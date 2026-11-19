// Keeps a copy of the page itself in the browser, so that a refresh while the server is asleep still
// shows the page (and its "dozed off" message) instead of the browser's "can't connect" screen.
const CACHE = "fovea-page-1";
const PAGE = ["/", "/style.css", "/app.js"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(PAGE)));
  self.skipWaiting();
});
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET" || !PAGE.includes(new URL(e.request.url).pathname)) return;
  e.respondWith(                       // the server's version when it is awake, the stored copy when not
    fetch(e.request).then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((cache) => cache.put(e.request, copy));
      return res;
    }).catch(() => caches.match(e.request)),
  );
});
