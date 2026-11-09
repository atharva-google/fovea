#!/usr/bin/env bash
# =============================================================================
#  Fovea backend → Google Cloud, guided deploy.
#
#      ./backend/gcp/deploy.sh
#
#  First run: checks your setup, asks a few questions, saves the answers to
#  backend/gcp/config.env, then deploys. Later runs reuse config.env
#  (edit it, or delete it to answer the questions again). Safe to re-run.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG="$HERE/config.env"
cd "$ROOT"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$*"; }
die()  { printf "\n  \033[31m✗ %s\033[0m\n\n" "$*"; exit 1; }
ask()  { # ask VAR "Question" "default"
  local reply; read -r -p "  $2 [$3]: " reply || true
  printf -v "$1" '%s' "${reply:-$3}"
}

bold "Fovea backend: Google Cloud deploy"
echo

# ----------------------------------------------------------------------------- 0. checks
bold "Checking your setup"
command -v gcloud >/dev/null 2>&1 || die "The gcloud CLI is not installed.
    Install it from https://cloud.google.com/sdk/docs/install , then run this script again."
ok "gcloud is installed"
ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -n1)"
[[ -n "$ACCOUNT" ]] || die "You are not logged in. Run:  gcloud auth login"
ok "logged in as $ACCOUNT"
command -v python3 >/dev/null 2>&1 || die "python3 is needed (it's used to write small config files)."

# ----------------------------------------------------------------------------- 1. config
if [[ ! -f "$CONFIG" ]]; then
  echo
  bold "A few questions (press Enter to accept the [default])"
  ask PROJECT_ID "Google Cloud project ID" "$(gcloud config get-value project 2>/dev/null || true)"
  [[ -n "$PROJECT_ID" ]] || die "A project ID is required. Create one at https://console.cloud.google.com/projectcreate"
  ask REGION "Region (must offer Cloud Run L4 GPUs)" "us-central1"
  ask BUCKET "Storage bucket name" "${PROJECT_ID}-fovea"
  ask HF_REPO "Hugging Face repo with the model weights (blank = official sources)" ""
  ask GPU_QUOTA "How many L4 GPUs may Cloud Run jobs use at once in $REGION (your quota)?" "1"
  if [[ "$GPU_QUOTA" -ge 2 ]]; then
    SPLIT_STEPS=true;  MAX_CONCURRENT_JOBS=$(( GPU_QUOTA / 2 ))
  else
    SPLIT_STEPS=false; MAX_CONCURRENT_JOBS=1
  fi
  ask AUTH_MODE "Who may call the API? 'iam' = only your frontend's service account, 'api_key' = anyone with a key" "iam"
  FRONTEND_SA=""; API_KEY=""
  if [[ "$AUTH_MODE" == "api_key" ]]; then
    API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  else
    ask FRONTEND_SA "Frontend service account email (blank = add later)" ""
  fi
  ask UPLOAD_ORIGINS "Website address(es) your frontend runs on, comma separated" "http://localhost:3000"
  ask DEMO_MODE "Demo mode: limit uploads, video length and daily jobs? (true/false)" "true"

  cat > "$CONFIG" <<EOF
# Fovea backend deploy settings, written by deploy.sh. Edit and re-run deploy.sh to apply.

# ---- Google Cloud
PROJECT_ID=$PROJECT_ID
REGION=$REGION
BUCKET=$BUCKET
RETENTION_DAYS=7                # job files are auto-deleted after this many days
HF_REPO=$HF_REPO

# ---- Who may call the API
AUTH_MODE=$AUTH_MODE            # iam | api_key
FRONTEND_SA=$FRONTEND_SA
API_KEY=$API_KEY
UPLOAD_ORIGINS=$UPLOAD_ORIGINS  # browser origins allowed to upload/download files

# ---- Limits
DEMO_MODE=$DEMO_MODE
MAX_UPLOAD_MB=500
MAX_DURATION_S=300
DAILY_JOBS_PER_USER=3
MAX_CONCURRENT_JOBS=$MAX_CONCURRENT_JOBS   # jobs processing at the same time

# ---- GPU worker
SPLIT_STEPS=$SPLIT_STEPS        # true: each step is its own GPU task (needs >= 2 GPUs of quota)
MAX_SHARDS=1                    # >1 splits long videos across more GPUs
GPU_TYPE=nvidia-l4
WORKER_CPU=8
WORKER_MEMORY=32Gi
TASK_TIMEOUT=30m
EOF
  ok "saved your answers to backend/gcp/config.env"
fi

set -a; source "$CONFIG"; set +a
: "${PROJECT_ID:?missing in config.env}" "${REGION:?missing in config.env}" "${BUCKET:?missing in config.env}"

G="--project=$PROJECT_ID --quiet"
TAG="$(date +%Y%m%d-%H%M%S)"
AR_REPO=fovea
API_SERVICE=fovea-api
WORKER_JOB=fovea-worker
WORKFLOW=fovea-pipeline
API_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/api:$TAG"
WORKER_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$AR_REPO/worker:$TAG"
API_SA="fovea-api@$PROJECT_ID.iam.gserviceaccount.com"
WORKER_SA="fovea-worker@$PROJECT_ID.iam.gserviceaccount.com"
WORKFLOW_SA="fovea-workflow@$PROJECT_ID.iam.gserviceaccount.com"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

echo
BILLING="$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)' 2>/dev/null || true)"
if [[ "$BILLING" == "False" ]]; then
  die "Billing is not enabled for $PROJECT_ID. Enable it at
    https://console.cloud.google.com/billing/linkedaccount?project=$PROJECT_ID"
elif [[ "$BILLING" == "True" ]]; then ok "billing is enabled"
else warn "could not check billing; make sure it's enabled for $PROJECT_ID"; fi

echo
bold "Before we start"
echo "  Cloud Run GPU jobs need GPU quota in $REGION. If you haven't requested it, do it here:"
echo "    https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID  (search: 'Cloud Run' + 'L4')"
echo "  Layout: SPLIT_STEPS=$SPLIT_STEPS, up to $MAX_CONCURRENT_JOBS job(s) at once."
read -r -p "  Deploy now? [y/N]: " GO || true
[[ "$GO" =~ ^[Yy]$ ]] || { echo "  Cancelled."; exit 0; }

step() { echo; bold "[$1/8] $2"; }

# ----------------------------------------------------------------------------- 2. deploy
step 1 "Turning on the Google Cloud services Fovea uses"
gcloud services enable run.googleapis.com workflows.googleapis.com workflowexecutions.googleapis.com \
  firestore.googleapis.com storage.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com iamcredentials.googleapis.com $G
ok "services enabled"

step 2 "Storage bucket gs://$BUCKET"
gcloud storage buckets describe "gs://$BUCKET" $G >/dev/null 2>&1 || \
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access $G
cat > "$TMP/lifecycle.json" <<EOF
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": ${RETENTION_DAYS:-7}, "matchesPrefix": ["users/"]}}]}
EOF
python3 - "$TMP/cors.json" <<'PY'
import json, os, sys
origins = [o.strip() for o in os.environ.get("UPLOAD_ORIGINS", "").split(",") if o.strip()]
json.dump([{"origin": origins, "method": ["PUT", "GET", "HEAD"],
            "responseHeader": ["Content-Type", "x-goog-content-length-range"], "maxAgeSeconds": 3600}],
          open(sys.argv[1], "w"))
PY
gcloud storage buckets update "gs://$BUCKET" --lifecycle-file="$TMP/lifecycle.json" --cors-file="$TMP/cors.json" $G
ok "bucket ready (files auto-delete after ${RETENTION_DAYS:-7} days)"

step 3 "Firestore database (job records)"
gcloud firestore databases describe --database="(default)" $G >/dev/null 2>&1 || \
  gcloud firestore databases create --location="$REGION" --type=firestore-native $G
ok "Firestore ready"

step 4 "Service accounts and permissions"
for name in fovea-api fovea-worker fovea-workflow; do
  gcloud iam service-accounts describe "$name@$PROJECT_ID.iam.gserviceaccount.com" $G >/dev/null 2>&1 || \
    gcloud iam service-accounts create "$name" --display-name="$name" $G
done
bucket_role()  { gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$1" --role="$2" $G >/dev/null; }
project_role() { gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$1" --role="$2" --condition=None $G >/dev/null; }
bucket_role  "$API_SA"      roles/storage.objectAdmin
bucket_role  "$WORKER_SA"   roles/storage.objectAdmin
bucket_role  "$WORKFLOW_SA" roles/storage.objectViewer
project_role "$API_SA"      roles/datastore.user
project_role "$API_SA"      roles/workflows.invoker
project_role "$WORKER_SA"   roles/datastore.user
project_role "$WORKFLOW_SA" roles/run.developer      # run the worker job with per-step arguments
project_role "$WORKFLOW_SA" roles/logging.logWriter
gcloud iam service-accounts add-iam-policy-binding "$API_SA" --member="serviceAccount:$API_SA" \
  --role=roles/iam.serviceAccountTokenCreator $G >/dev/null   # lets the API sign upload/download links
gcloud iam service-accounts add-iam-policy-binding "$WORKER_SA" --member="serviceAccount:$WORKFLOW_SA" \
  --role=roles/iam.serviceAccountUser $G >/dev/null
ok "permissions set"

step 5 "Building the API and GPU worker images (first time: ~20-30 min)"
gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" $G >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$REGION" $G
# Newer projects run Cloud Build as the Compute Engine default service account, which often
# lacks the permissions to read the uploaded source, write logs and push images.
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
BUILD_SA="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"
for role in roles/artifactregistry.writer roles/logging.logWriter roles/storage.objectViewer; do
  project_role "$BUILD_SA" "$role"
done
gcloud builds submit --config backend/gcp/cloudbuild.yaml \
  --substitutions="_API_IMAGE=$API_IMAGE,_WORKER_IMAGE=$WORKER_IMAGE,_HF_REPO=${HF_REPO:-}" $G .
ok "images built"

step 6 "GPU worker job ($WORKER_JOB)"
gcloud beta run jobs deploy "$WORKER_JOB" --image="$WORKER_IMAGE" --region="$REGION" \
  --service-account="$WORKER_SA" --gpu=1 --gpu-type="${GPU_TYPE:-nvidia-l4}" \
  --cpu="${WORKER_CPU:-8}" --memory="${WORKER_MEMORY:-32Gi}" \
  --task-timeout="${TASK_TIMEOUT:-30m}" --max-retries=1 --tasks=1 \
  --set-env-vars="MODE=gcp,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,BUCKET=$BUCKET" $G
ok "worker job ready"

step 7 "API service ($API_SERVICE)"
cat > "$TMP/api-env.yaml" <<EOF
MODE: "gcp"
GOOGLE_CLOUD_PROJECT: "$PROJECT_ID"
REGION: "$REGION"
BUCKET: "$BUCKET"
WORKFLOW_NAME: "$WORKFLOW"
WORKER_JOB_NAME: "$WORKER_JOB"
WORKFLOW_SA_EMAIL: "$WORKFLOW_SA"
AUTH_MODE: "${AUTH_MODE:-iam}"
API_KEY: "${API_KEY:-}"
DEMO_MODE: "${DEMO_MODE:-true}"
MAX_UPLOAD_MB: "${MAX_UPLOAD_MB:-500}"
MAX_DURATION_S: "${MAX_DURATION_S:-300}"
DAILY_JOBS_PER_USER: "${DAILY_JOBS_PER_USER:-3}"
MAX_CONCURRENT_JOBS: "${MAX_CONCURRENT_JOBS:-1}"
SPLIT_STEPS: "${SPLIT_STEPS:-false}"
MAX_SHARDS: "${MAX_SHARDS:-1}"
EOF
if [[ "${AUTH_MODE:-iam}" == "api_key" ]]; then AUTH_FLAG=--allow-unauthenticated; else AUTH_FLAG=--no-allow-unauthenticated; fi
gcloud run deploy "$API_SERVICE" --image="$API_IMAGE" --region="$REGION" --service-account="$API_SA" \
  --cpu=1 --memory=1Gi --env-vars-file="$TMP/api-env.yaml" $AUTH_FLAG $G
API_URL="$(gcloud run services describe "$API_SERVICE" --region="$REGION" --format='value(status.url)' $G)"
gcloud run services update "$API_SERVICE" --region="$REGION" --update-env-vars="SELF_URL=$API_URL" $G
gcloud run services add-iam-policy-binding "$API_SERVICE" --region="$REGION" \
  --member="serviceAccount:$WORKFLOW_SA" --role=roles/run.invoker $G >/dev/null
if [[ -n "${FRONTEND_SA:-}" ]]; then
  gcloud run services add-iam-policy-binding "$API_SERVICE" --region="$REGION" \
    --member="serviceAccount:$FRONTEND_SA" --role=roles/run.invoker $G >/dev/null
fi
ok "API live at $API_URL"

step 8 "Pipeline workflow ($WORKFLOW)"
gcloud workflows deploy "$WORKFLOW" --source=backend/gcp/pipeline.yaml --location="$REGION" \
  --service-account="$WORKFLOW_SA" $G
ok "workflow deployed"

# ----------------------------------------------------------------------------- 3. done
echo
bold "Done! Your backend is live."
echo
echo "  API URL:   $API_URL"
echo "  Bucket:    gs://$BUCKET"
echo
bold "Test it"
if [[ "${AUTH_MODE:-iam}" == "api_key" ]]; then
  echo "  curl -H 'X-API-Key: $API_KEY' $API_URL/v1/config"
else
  echo "  curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" $API_URL/v1/config"
fi
echo
bold "Give your frontend"
echo "  BACKEND_URL=$API_URL"
if [[ "${AUTH_MODE:-iam}" == "api_key" ]]; then
  echo "  BACKEND_API_KEY=$API_KEY      (send as the X-API-Key header; keep it secret)"
else
  echo "  The frontend must run as a service account with access to the API."
  [[ -z "${FRONTEND_SA:-}" ]] && echo "  → Put its email in FRONTEND_SA in backend/gcp/config.env and re-run this script."
fi
echo
