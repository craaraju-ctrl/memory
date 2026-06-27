//! # Error Types — API Error Handling
//!
//! Provides `AppError` — a centralized error type for the HTTP API
//! that renders as structured JSON error responses via axum's `IntoResponse`.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// A unified error type for the HTTP API layer.
#[derive(Debug)]
pub enum AppError {
    Database(String),
    NotFound(String),
    BadRequest(String),
    Internal(String),
}

impl AppError {
    fn status_code(&self) -> StatusCode {
        match self {
            AppError::Database(_) => StatusCode::INTERNAL_SERVER_ERROR,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let message = match &self {
            AppError::Database(msg)
            | AppError::NotFound(msg)
            | AppError::BadRequest(msg)
            | AppError::Internal(msg) => msg,
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

impl From<String> for AppError {
    fn from(msg: String) -> Self {
        AppError::Internal(msg)
    }
}

impl From<&str> for AppError {
    fn from(msg: &str) -> Self {
        AppError::Internal(msg.to_string())
    }
}

impl From<rusqlite::Error> for AppError {
    fn from(e: rusqlite::Error) -> Self {
        AppError::Database(e.to_string())
    }
}
