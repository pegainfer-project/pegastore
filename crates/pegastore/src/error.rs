//! Errors are for programs first: a closed `ErrorKind`, a retryability
//! status, and key/value context for humans reading logs.

use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Key does not exist.
    NotFound,
    /// Slot exists but currently has no replica anywhere. Recompute or replay.
    Evicted,
    /// Write-once conflict: the slot already has bytes (or a write in flight).
    AlreadyExists,
    /// Same key, different `ObjectSpec`.
    SpecMismatch,
    /// `Strict` placement cannot be satisfied, or the pool is full of `Explicit` objects.
    NoSpace,
    /// The backend's `Capability` says no.
    Unsupported,
    /// Malformed request: iov out of bounds, length mismatch, unknown slot, ...
    InvalidInput,
    /// Daemon / metaserver / remote peer unreachable. Never a cache miss.
    Unavailable,
    Unexpected,
}

impl ErrorKind {
    pub const fn default_status(self) -> ErrorStatus {
        match self {
            ErrorKind::NoSpace | ErrorKind::Unavailable => ErrorStatus::Temporary,
            _ => ErrorStatus::Permanent,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            ErrorKind::NotFound => "NotFound",
            ErrorKind::Evicted => "Evicted",
            ErrorKind::AlreadyExists => "AlreadyExists",
            ErrorKind::SpecMismatch => "SpecMismatch",
            ErrorKind::NoSpace => "NoSpace",
            ErrorKind::Unsupported => "Unsupported",
            ErrorKind::InvalidInput => "InvalidInput",
            ErrorKind::Unavailable => "Unavailable",
            ErrorKind::Unexpected => "Unexpected",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ErrorStatus {
    /// Retrying the same request will not help.
    Permanent,
    /// Retrying (with backoff) may succeed.
    Temporary,
}

pub struct Error {
    kind: ErrorKind,
    status: ErrorStatus,
    message: String,
    operation: Option<&'static str>,
    context: Vec<(&'static str, String)>,
    source: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            status: kind.default_status(),
            message: message.into(),
            operation: None,
            context: Vec::new(),
            source: None,
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn status(&self) -> ErrorStatus {
        self.status
    }

    pub fn is_temporary(&self) -> bool {
        self.status == ErrorStatus::Temporary
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn operation(&self) -> Option<&'static str> {
        self.operation
    }

    pub fn context(&self) -> &[(&'static str, String)] {
        &self.context
    }

    pub fn with_operation(mut self, operation: &'static str) -> Self {
        if self.operation.is_none() {
            self.operation = Some(operation);
        }
        self
    }

    pub fn with_context(mut self, key: &'static str, value: impl ToString) -> Self {
        self.context.push((key, value.to_string()));
        self
    }

    pub fn set_temporary(mut self) -> Self {
        self.status = ErrorStatus::Temporary;
        self
    }

    pub fn set_permanent(mut self) -> Self {
        self.status = ErrorStatus::Permanent;
        self
    }

    pub fn set_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)?;
        if self.status == ErrorStatus::Temporary {
            f.write_str(" (temporary)")?;
        }
        if let Some(op) = self.operation {
            write!(f, " at {op}")?;
        }
        if !self.message.is_empty() {
            write!(f, " => {}", self.message)?;
        }
        if !self.context.is_empty() {
            f.write_str(" {")?;
            for (i, (k, v)) in self.context.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{k}: {v}")?;
            }
            f.write_str("}")?;
        }
        if let Some(src) = &self.source {
            write!(f, ": {src}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_deref().map(|e| e as _)
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
