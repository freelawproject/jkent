-- index: idx_archived_files_hash
CREATE INDEX idx_archived_files_hash ON archived_files (content_hash);

-- index: idx_archived_files_request
CREATE INDEX idx_archived_files_request ON archived_files (request_id);

-- index: idx_compression_dicts_continuation
CREATE INDEX idx_compression_dicts_continuation ON compression_dicts (continuation);

-- index: idx_errors_request
CREATE INDEX idx_errors_request ON errors (request_id);

-- index: idx_errors_type
CREATE INDEX idx_errors_type ON errors (error_type);

-- index: idx_incidental_requests_parent
CREATE INDEX idx_incidental_requests_parent ON incidental_requests (parent_request_id);

-- index: idx_incidental_requests_storage
CREATE INDEX idx_incidental_requests_storage ON incidental_requests (storage_id);

-- index: idx_irs_content_md5
CREATE INDEX idx_irs_content_md5 ON incidental_request_storage (content_md5);

-- index: idx_requests_cache_key
CREATE INDEX idx_requests_cache_key ON requests (cache_key);

-- index: idx_requests_compression_dict
CREATE INDEX idx_requests_compression_dict ON requests (compression_dict_id);

-- index: idx_requests_continuation
CREATE INDEX idx_requests_continuation ON requests (continuation);

-- index: idx_requests_parent
CREATE INDEX idx_requests_parent ON requests (parent_request_id);

-- index: idx_requests_response_status_code
CREATE INDEX idx_requests_response_status_code ON requests (response_status_code);

-- index: idx_requests_speculation
CREATE INDEX idx_requests_speculation ON requests (speculation_tracking_id);

-- index: idx_requests_status_priority
CREATE INDEX idx_requests_status_priority ON requests (status, priority, queue_counter);

-- index: idx_results_request
CREATE INDEX idx_results_request ON results (request_id);

-- index: idx_results_type
CREATE INDEX idx_results_type ON results (result_type);

-- index: idx_speculation_tracking_func
CREATE INDEX idx_speculation_tracking_func ON speculation_tracking (func_name);

-- table: archived_files
CREATE TABLE archived_files (
	id INTEGER NOT NULL,
	request_id INTEGER NOT NULL,
	file_path VARCHAR NOT NULL,
	original_url VARCHAR NOT NULL,
	expected_type VARCHAR,
	file_size INTEGER,
	content_hash VARCHAR,
	created_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	PRIMARY KEY (id),
	FOREIGN KEY(request_id) REFERENCES requests (id) ON DELETE CASCADE
);

-- table: compression_dicts
CREATE TABLE compression_dicts (
	id INTEGER NOT NULL,
	continuation VARCHAR NOT NULL,
	version INTEGER NOT NULL,
	dictionary_data BLOB NOT NULL,
	sample_count INTEGER NOT NULL,
	created_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	PRIMARY KEY (id),
	UNIQUE (continuation, version)
);

-- table: errors
CREATE TABLE errors (
	id INTEGER NOT NULL,
	request_id INTEGER,
	error_type INTEGER NOT NULL,
	error_class VARCHAR NOT NULL,
	message VARCHAR NOT NULL,
	request_url VARCHAR NOT NULL,
	context_json VARCHAR,
	selector VARCHAR,
	selector_type INTEGER,
	expected_min INTEGER,
	expected_max INTEGER,
	actual_count INTEGER,
	model_name VARCHAR,
	validation_errors_json VARCHAR,
	failed_doc_json VARCHAR,
	status_code INTEGER,
	timeout_seconds FLOAT,
	traceback VARCHAR,
	created_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	PRIMARY KEY (id),
	CONSTRAINT ck_errors_error_type CHECK (error_type IN (1, 2, 3, 4, 5)),
	CONSTRAINT ck_errors_selector_type CHECK (selector_type IN (1, 2)),
	CONSTRAINT ck_errors_context_json_valid CHECK (json_valid(context_json)),
	CONSTRAINT ck_errors_validation_errors_json_valid CHECK (json_valid(validation_errors_json)),
	CONSTRAINT ck_errors_failed_doc_json_valid CHECK (json_valid(failed_doc_json)),
	FOREIGN KEY(request_id) REFERENCES requests (id) ON DELETE CASCADE
);

-- table: incidental_request_storage
CREATE TABLE incidental_request_storage (
	id INTEGER NOT NULL,
	resource_type VARCHAR NOT NULL,
	url VARCHAR NOT NULL,
	method VARCHAR NOT NULL,
	body BLOB,
	status_code INTEGER,
	response_headers_json VARCHAR,
	content_compressed BLOB,
	content_size_original INTEGER,
	content_size_compressed INTEGER,
	compression_dict_id INTEGER,
	failure_reason VARCHAR,
	content_md5 BLOB,
	PRIMARY KEY (id),
	CONSTRAINT ck_incidental_request_storage_response_headers_json_valid CHECK (json_valid(response_headers_json)),
	FOREIGN KEY(compression_dict_id) REFERENCES compression_dicts (id)
);

-- table: incidental_requests
CREATE TABLE incidental_requests (
	id INTEGER NOT NULL,
	parent_request_id INTEGER NOT NULL,
	url VARCHAR NOT NULL,
	headers_json VARCHAR,
	started_at_ns INTEGER,
	completed_at_ns INTEGER,
	from_cache BOOLEAN,
	created_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	storage_id INTEGER,
	PRIMARY KEY (id),
	CONSTRAINT ck_incidental_requests_headers_json_valid CHECK (json_valid(headers_json)),
	FOREIGN KEY(parent_request_id) REFERENCES requests (id) ON DELETE CASCADE,
	FOREIGN KEY(storage_id) REFERENCES incidental_request_storage (id)
);

-- table: requests
CREATE TABLE requests (
	id INTEGER NOT NULL,
	status INTEGER DEFAULT 1 NOT NULL,
	priority INTEGER DEFAULT 9 NOT NULL,
	queue_counter INTEGER NOT NULL,
	request_type INTEGER DEFAULT 1 NOT NULL,
	method INTEGER NOT NULL,
	url VARCHAR NOT NULL,
	headers_json VARCHAR,
	cookies_json VARCHAR,
	body BLOB,
	continuation VARCHAR NOT NULL,
	current_location VARCHAR DEFAULT '' NOT NULL,
	accumulated_data_json VARCHAR,
	permanent_json VARCHAR,
	deduplication_key VARCHAR,
	cache_key BLOB,
	expected_type VARCHAR,
	bypass_rate_limit BOOLEAN DEFAULT 0 NOT NULL,
	created_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	started_at DATETIME,
	completed_at DATETIME,
	retry_count INTEGER DEFAULT 0 NOT NULL,
	cumulative_backoff FLOAT DEFAULT (0.0) NOT NULL,
	last_error VARCHAR,
	parent_request_id INTEGER,
	is_speculative BOOLEAN DEFAULT 0 NOT NULL,
	speculation_tracking_id INTEGER,
	speculative_index INTEGER,
	response_status_code INTEGER,
	response_headers_json VARCHAR,
	response_url VARCHAR,
	content_compressed BLOB,
	content_size_original INTEGER,
	content_size_compressed INTEGER,
	compression_dict_id INTEGER,
	response_created_at DATETIME,
	speculation_outcome INTEGER,
	via_json VARCHAR,
	verify VARCHAR,
	timeout_json VARCHAR,
	json_data VARCHAR,
	files_json VARCHAR,
	auth_json VARCHAR,
	allow_redirects BOOLEAN DEFAULT 1 NOT NULL,
	proxies_json VARCHAR,
	stream BOOLEAN DEFAULT 0 NOT NULL,
	cert_json VARCHAR,
	archive_hash_header VARCHAR,
	reseedable BOOLEAN,
	preresolved BOOLEAN DEFAULT 0 NOT NULL,
	PRIMARY KEY (id),
	CONSTRAINT uq_requests_dedup_key UNIQUE (deduplication_key) ON CONFLICT IGNORE,
	CONSTRAINT ck_requests_status CHECK (status IN (1, 2, 3, 4, 5, 6)),
	CONSTRAINT ck_requests_request_type CHECK (request_type IN (1, 2, 3)),
	CONSTRAINT ck_requests_method CHECK (method IN (1, 2, 3, 4, 5, 6, 7)),
	CONSTRAINT ck_requests_speculation_outcome CHECK (speculation_outcome IN (1, 2, 3)),
	CONSTRAINT ck_requests_headers_json_valid CHECK (json_valid(headers_json)),
	CONSTRAINT ck_requests_cookies_json_valid CHECK (json_valid(cookies_json)),
	CONSTRAINT ck_requests_accumulated_data_json_valid CHECK (json_valid(accumulated_data_json)),
	CONSTRAINT ck_requests_permanent_json_valid CHECK (json_valid(permanent_json)),
	CONSTRAINT ck_requests_response_headers_json_valid CHECK (json_valid(response_headers_json)),
	CONSTRAINT ck_requests_via_json_valid CHECK (json_valid(via_json)),
	CONSTRAINT ck_requests_timeout_json_valid CHECK (json_valid(timeout_json)),
	CONSTRAINT ck_requests_files_json_valid CHECK (json_valid(files_json)),
	CONSTRAINT ck_requests_auth_json_valid CHECK (json_valid(auth_json)),
	CONSTRAINT ck_requests_proxies_json_valid CHECK (json_valid(proxies_json)),
	CONSTRAINT ck_requests_cert_json_valid CHECK (json_valid(cert_json)),
	FOREIGN KEY(parent_request_id) REFERENCES requests (id) ON DELETE CASCADE,
	FOREIGN KEY(speculation_tracking_id) REFERENCES speculation_tracking (id),
	FOREIGN KEY(compression_dict_id) REFERENCES compression_dicts (id)
);

-- table: results
CREATE TABLE results (
	id INTEGER NOT NULL,
	request_id INTEGER NOT NULL,
	result_type VARCHAR NOT NULL,
	data_json VARCHAR NOT NULL,
	is_valid BOOLEAN DEFAULT 1 NOT NULL,
	validation_errors_json VARCHAR,
	created_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	PRIMARY KEY (id),
	CONSTRAINT ck_results_data_json_valid CHECK (json_valid(data_json)),
	CONSTRAINT ck_results_validation_errors_json_valid CHECK (json_valid(validation_errors_json)),
	FOREIGN KEY(request_id) REFERENCES requests (id) ON DELETE CASCADE
);

-- table: run_metadata
CREATE TABLE run_metadata (
	id INTEGER NOT NULL,
	scraper_name VARCHAR NOT NULL,
	scraper_version VARCHAR,
	status INTEGER DEFAULT 1 NOT NULL,
	created_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	started_at DATETIME,
	ended_at DATETIME,
	error_message VARCHAR,
	params_json VARCHAR,
	seed_params_json VARCHAR,
	jitter FLOAT NOT NULL,
	num_workers INTEGER NOT NULL,
	max_backoff_time FLOAT NOT NULL,
	speculation_config_json VARCHAR,
	browser_config_json VARCHAR,
	browser_cookies_json VARCHAR,
	PRIMARY KEY (id),
	CONSTRAINT run_metadata_single_row CHECK (id = 1),
	CONSTRAINT ck_run_metadata_status CHECK (status IN (1, 2, 3, 4, 5)),
	CONSTRAINT ck_run_metadata_params_json_valid CHECK (json_valid(params_json)),
	CONSTRAINT ck_run_metadata_seed_params_json_valid CHECK (json_valid(seed_params_json)),
	CONSTRAINT ck_run_metadata_speculation_config_json_valid CHECK (json_valid(speculation_config_json)),
	CONSTRAINT ck_run_metadata_browser_config_json_valid CHECK (json_valid(browser_config_json)),
	CONSTRAINT ck_run_metadata_browser_cookies_json_valid CHECK (json_valid(browser_cookies_json))
);

-- table: schema_info
CREATE TABLE schema_info (
	id INTEGER NOT NULL,
	version INTEGER NOT NULL,
	applied_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	PRIMARY KEY (id)
);

-- table: speculation_tracking
CREATE TABLE speculation_tracking (
	id INTEGER NOT NULL,
	func_name VARCHAR NOT NULL,
	highest_successful_id INTEGER DEFAULT 0 NOT NULL,
	consecutive_failures INTEGER DEFAULT 0 NOT NULL,
	current_ceiling INTEGER DEFAULT 0 NOT NULL,
	stopped BOOLEAN DEFAULT 0 NOT NULL,
	param_index INTEGER DEFAULT 0 NOT NULL,
	template_json VARCHAR,
	updated_at DATETIME DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')),
	seed_value_json VARCHAR,
	PRIMARY KEY (id),
	CONSTRAINT ck_speculation_tracking_template_json_valid CHECK (json_valid(template_json)),
	CONSTRAINT ck_speculation_tracking_seed_value_json_valid CHECK (json_valid(seed_value_json)),
	UNIQUE (func_name)
);
