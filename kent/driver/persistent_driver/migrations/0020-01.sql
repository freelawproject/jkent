-- v19 → v20: Round-trip the rest of HTTPRequestParams.
--
-- Pre-v20 the persistent queue silently dropped most non-default
-- HTTPRequestParams fields (timeout, json, files, auth, allow_redirects,
-- proxies, stream, cert) plus the Request-level `archive_hash_header`.
-- That is why scrapers like Nevada Supreme Court, which set
-- `timeout=360.0` on archive downloads, saw their timeouts ignored.
ALTER TABLE requests ADD COLUMN timeout_json TEXT;
ALTER TABLE requests ADD COLUMN json_data TEXT;
ALTER TABLE requests ADD COLUMN files_json TEXT;
ALTER TABLE requests ADD COLUMN auth_json TEXT;
ALTER TABLE requests ADD COLUMN allow_redirects BOOLEAN DEFAULT 1;
ALTER TABLE requests ADD COLUMN proxies_json TEXT;
ALTER TABLE requests ADD COLUMN stream BOOLEAN DEFAULT 0;
ALTER TABLE requests ADD COLUMN cert_json TEXT;
ALTER TABLE requests ADD COLUMN archive_hash_header TEXT;
