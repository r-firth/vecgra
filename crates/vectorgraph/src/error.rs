use std::fmt;
use std::io;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Corrupt(String),
    InvalidArgument(String),
    NotFound(&'static str, u64),
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
