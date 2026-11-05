// SPDX-License-Identifier: 0BSD

use std::fmt;

/// Stable error categories shared by the Python and C ABI boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum RusticolErrorKind {
    InvalidArgument,
    Artifact,
    Security,
    Integrity,
    Compatibility,
    Serialization,
    Evaluation,
    Selector,
    ModelParameter,
    UnsupportedPrecision,
    UnsupportedRuntimeCapability,
    Internal,
}

/// A Python-independent runtime error with a machine-readable category.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RusticolError {
    kind: RusticolErrorKind,
    message: String,
}

pub type RusticolResult<T> = Result<T, RusticolError>;

impl RusticolError {
    pub fn new(message: impl Into<String>) -> Self {
        Self::internal(message)
    }

    pub fn with_kind(kind: RusticolErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::InvalidArgument, message)
    }

    pub fn artifact(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::Artifact, message)
    }

    pub fn security(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::Security, message)
    }

    pub fn integrity(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::Integrity, message)
    }

    pub fn compatibility(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::Compatibility, message)
    }

    pub fn serialization(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::Serialization, message)
    }

    pub fn evaluation(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::Evaluation, message)
    }

    pub fn selector(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::Selector, message)
    }

    pub fn model_parameter(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::ModelParameter, message)
    }

    pub fn unsupported_precision(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::UnsupportedPrecision, message)
    }

    pub fn unsupported_runtime_capability(
        capability: impl AsRef<str>,
        detail: impl AsRef<str>,
    ) -> Self {
        Self::with_kind(
            RusticolErrorKind::UnsupportedRuntimeCapability,
            format!(
                "unsupported runtime capability {:?}: {}",
                capability.as_ref(),
                detail.as_ref()
            ),
        )
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::with_kind(RusticolErrorKind::Internal, message)
    }

    pub fn kind(&self) -> RusticolErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for RusticolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for RusticolError {}
