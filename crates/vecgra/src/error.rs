use std::fmt;
use std::io;

/// Result type returned by Vecgra operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned by storage, validation, and graph operations.
#[derive(Debug)]
pub enum Error {
    /// Filesystem or stream I/O failed.
    Io(io::Error),
    /// The database file violates a validated format invariant.
    Corrupt(String),
    /// An argument is invalid for the database or operation.
    InvalidArgument(String),
    /// A requested graph record does not exist.
    NotFound(&'static str, u64),
    /// A write conflicts with the current graph state.
    Conflict(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Corrupt(message) => write!(f, "corrupt database: {message}"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::NotFound(kind, id) => write!(f, "{kind} {id} was not found"),
            Self::Conflict(message) => write!(f, "transaction conflict: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
