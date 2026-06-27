//! # Resilience Module
//!
//! Provides reusable resilience components:
//! - `CircuitBreaker` — Prevents cascading failures
//! - `ResilientClient` — Combines Circuit Breaker + Retry for HTTP calls
//!
//! These components are designed to be used across the crate and by external consumers.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{policies::ExponentialBackoff, RetryTransientMiddleware};

/// Circuit Breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// A reusable, thread-safe Circuit Breaker implementation.
#[derive(Debug)]
pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
    failure_count: Arc<Mutex<u32>>,
    last_failure_time: Arc<Mutex<Option<Instant>>>,
    failure_threshold: u32,
    reset_timeout: Duration,
}

impl CircuitBreaker {
    /// Create a new Circuit Breaker.
    ///
    /// - `failure_threshold`: Number of consecutive failures before opening the circuit.
    /// - `reset_timeout_secs`: Time in seconds after which to attempt recovery (Half-Open).
    pub fn new(failure_threshold: u32, reset_timeout_secs: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            failure_count: Arc::new(Mutex::new(0)),
            last_failure_time: Arc::new(Mutex::new(None)),
            failure_threshold,
            reset_timeout: Duration::from_secs(reset_timeout_secs),
        }
    }

    /// Check if a request is allowed to proceed.
    pub fn can_execute(&self) -> bool {
        let state = self.state.lock().unwrap();
        match *state {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                if let Some(last_failure) = *self.last_failure_time.lock().unwrap() {
                    if last_failure.elapsed() >= self.reset_timeout {
                        drop(state);
                        let mut state = self.state.lock().unwrap();
                        *state = CircuitState::HalfOpen;
                        return true;
                    }
                }
                false
            }
        }
    }

    /// Record a successful request.
    pub fn on_success(&self) {
        let mut state = self.state.lock().unwrap();
        let mut failure_count = self.failure_count.lock().unwrap();
        *state = CircuitState::Closed;
        *failure_count = 0;
    }

    /// Record a failed request.
    pub fn on_failure(&self) {
        let mut failure_count = self.failure_count.lock().unwrap();
        *failure_count += 1;

        let mut last_failure_time = self.last_failure_time.lock().unwrap();
        *last_failure_time = Some(Instant::now());

        if *failure_count >= self.failure_threshold {
            let mut state = self.state.lock().unwrap();
            *state = CircuitState::Open;
        }
    }

    /// Get the current state of the circuit breaker.
    pub fn get_state(&self) -> CircuitState {
        *self.state.lock().unwrap()
    }
}

/// A reusable resilient HTTP client that combines Circuit Breaker + Retry.
pub struct ResilientClient {
    client: ClientWithMiddleware,
    circuit_breaker: CircuitBreaker,
}

impl ResilientClient {
    /// Create a new Resilient HTTP Client.
    pub fn new(max_retries: u32, failure_threshold: u32, reset_timeout_secs: u64) -> Self {
        let reqwest_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build reqwest client");

        let retry_policy = ExponentialBackoff::builder().build_with_max_retries(max_retries);

        let client = ClientBuilder::new(reqwest_client)
            .with(RetryTransientMiddleware::new_with_policy(retry_policy))
            .build();

        let circuit_breaker = CircuitBreaker::new(failure_threshold, reset_timeout_secs);

        Self {
            client,
            circuit_breaker,
        }
    }

    /// Perform a resilient POST request with JSON body and custom headers.
    pub async fn post_json_with_headers<T, R>(
        &self,
        url: &str,
        body: &T,
        headers: &[(&str, &str)],
    ) -> Result<R, String>
    where
        T: serde::Serialize,
        R: serde::de::DeserializeOwned,
    {
        if !self.circuit_breaker.can_execute() {
            return Err("Circuit breaker is OPEN. Service temporarily unavailable.".to_string());
        }

        let mut builder = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(body).unwrap_or_default());

        for (key, value) in headers {
            builder = builder.header(*key, *value);
        }

        let result = builder.send().await;

        match result {
            Ok(response) if response.status().is_success() => {
                self.circuit_breaker.on_success();
                response.json::<R>().await.map_err(|e| e.to_string())
            }
            Ok(response) => {
                self.circuit_breaker.on_failure();
                Err(format!("HTTP error: {}", response.status()))
            }
            Err(e) => {
                self.circuit_breaker.on_failure();
                Err(e.to_string())
            }
        }
    }

    /// Perform a resilient POST request with JSON body.
    pub async fn post_json<T, R>(&self, url: &str, body: &T) -> Result<R, String>
    where
        T: serde::Serialize,
        R: serde::de::DeserializeOwned,
    {
        if !self.circuit_breaker.can_execute() {
            return Err("Circuit breaker is OPEN. Service temporarily unavailable.".to_string());
        }

        let result = self
            .client
            .post(url)
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(body).unwrap_or_default())
            .send()
            .await;

        match result {
            Ok(response) if response.status().is_success() => {
                self.circuit_breaker.on_success();
                response.json::<R>().await.map_err(|e| e.to_string())
            }
            Ok(response) => {
                self.circuit_breaker.on_failure();
                Err(format!("HTTP error: {}", response.status()))
            }
            Err(e) => {
                self.circuit_breaker.on_failure();
                Err(e.to_string())
            }
        }
    }
}
