use std::fmt;
use std::error::Error;

pub type VoxBoxResult<T> = Result<T, VoxBoxError>;

#[derive(Debug)]
pub enum VoxBoxError {
    /// LPC calculation error
    LPC(&'static str),
    /// Pitch calculation error
    Pitch(&'static str),
    /// Polynomial calculation error
    Polynomial(&'static str),
    /// Not enough workspace allocated
    Workspace,
}

impl fmt::Display for VoxBoxError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt.write_str(self.description())
    }
}

impl Error for VoxBoxError {
    fn description(&self) -> &str {
        use self::VoxBoxError::*;
        match *self {
            LPC(s) => s,
            Pitch(s) => s,
            Polynomial(s) => s,
            Workspace => "Not enough workspace allocated",
        }
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

