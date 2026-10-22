use std::io;
use std::path::{Path, PathBuf};

/// Databases opened recently, newest first. Kept as one path per line in a
/// small file so a later session can offer them again.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecentDatabases {
    paths: Vec<PathBuf>,
}

impl RecentDatabases {
    /// How many databases are remembered.
    pub const LIMIT: usize = 8;

    /// Where Studio keeps the list for the current user, when the platform
    /// has a configuration directory.
    pub fn default_file() -> Option<PathBuf> {
        dirs::config_dir().map(|directory| directory.join("vecgra").join("studio-recent"))
    }

    /// Reads a saved list. A missing or unreadable file is an empty list, and
    /// databases that no longer exist are dropped.
    pub fn load(file: &Path) -> Self {
        let mut recent = Self::default();
        let Ok(contents) = std::fs::read_to_string(file) else {
            return recent;
        };
        for line in contents
            .lines()
            .map(|line| line.trim_end_matches('\r'))
            .filter(|line| !line.is_empty())
        {
            let path = PathBuf::from(line);
            if recent.paths.len() < Self::LIMIT && !recent.paths.contains(&path) && path.is_file() {
                recent.paths.push(path);
            }
        }
        recent
    }

    /// Writes the list, creating its directory, and replaces the previous file
    /// in one rename so a concurrent reader never sees half a list.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory or file cannot be written.
    pub fn save(&self, file: &Path) -> io::Result<()> {
        if let Some(directory) = file.parent() {
            std::fs::create_dir_all(directory)?;
        }
        let mut contents = String::new();
        for path in &self.paths {
            if let Some(path) = path.to_str() {
                contents.push_str(path);
                contents.push('\n');
            }
        }
        let staging = file.with_extension("tmp");
        std::fs::write(&staging, contents)?;
        std::fs::rename(staging, file)
    }

    /// Moves a database to the front of the list. Paths that cannot be
    /// written back as one line of UTF-8 are not remembered.
    pub fn remember(&mut self, path: PathBuf) {
        if path.to_str().is_none_or(|text| text.contains(['\n', '\r'])) {
            return;
        }
        self.paths.retain(|existing| *existing != path);
        self.paths.insert(0, path);
        self.paths.truncate(Self::LIMIT);
    }

    pub fn forget(&mut self, path: &Path) {
        self.paths.retain(|existing| existing != path);
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
}

/// Searches and commands submitted from the command box, oldest first. Kept
/// as one per line in a small file so Up in a later session still reaches
/// them, as a shell's history does.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CommandHistory {
    entries: Vec<String>,
}

impl CommandHistory {
    /// How many entries are remembered.
    pub const LIMIT: usize = 100;

    /// Where Studio keeps the history for the current user, when the
    /// platform has a configuration directory.
    pub fn default_file() -> Option<PathBuf> {
        dirs::config_dir().map(|directory| directory.join("vecgra").join("studio-history"))
    }

    /// Reads a saved history, keeping its newest entries. A missing or
    /// unreadable file is an empty history.
    pub fn load(file: &Path) -> Self {
        let mut history = Self::default();
        let Ok(contents) = std::fs::read_to_string(file) else {
            return history;
        };
        for line in contents.lines() {
            history.remember(line.trim_end_matches('\r'));
        }
        history
    }

    /// Writes the history, creating its directory, and replaces the previous
    /// file in one rename so a concurrent reader never sees half of it.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory or file cannot be written.
    pub fn save(&self, file: &Path) -> io::Result<()> {
        if let Some(directory) = file.parent() {
            std::fs::create_dir_all(directory)?;
        }
        let mut contents = String::new();
        for entry in &self.entries {
            contents.push_str(entry);
            contents.push('\n');
        }
        let staging = file.with_extension("tmp");
        std::fs::write(&staging, contents)?;
        std::fs::rename(staging, file)
    }

    /// Adds an entry as the newest, unless it is blank, spans lines, or
    /// repeats the newest entry. Returns whether the history changed.
    pub fn remember(&mut self, entry: &str) -> bool {
        let entry = entry.trim();
        if entry.is_empty()
            || entry.contains(['\n', '\r'])
            || self.entries.last().is_some_and(|last| last == entry)
        {
            return false;
        }
        if self.entries.len() == Self::LIMIT {
            self.entries.remove(0);
        }
        self.entries.push(entry.to_string());
        true
    }

    /// Entries, oldest first.
    pub fn entries(&self) -> &[String] {
        &self.entries
    }
}

/// A path as a person would write it, with the home directory as `~`.
pub fn display_path(path: &Path) -> String {
    dirs::home_dir()
        .and_then(|home| path.strip_prefix(home).ok())
        .map_or_else(
            || path.display().to_string(),
            |relative| Path::new("~").join(relative).display().to_string(),
        )
}
