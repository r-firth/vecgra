//! Readable ages for date-like properties: imported records keep ISO 8601
//! dates and millisecond timestamps (`created_at`, `created_at_ms`), which
//! say when something happened only after some arithmetic.

use crate::PropertyValue;

/// Millisecond timestamps outside 2000–2100 are more likely counts or IDs.
const PLAUSIBLE_MS: std::ops::Range<i64> = 946_684_800_000..4_102_444_800_000;

/// A hint to show beside a date-like property, relative to `now_ms` (Unix
/// milliseconds): the age of an ISO 8601 date-time ("3 days ago"), or the
/// UTC date and age of a millisecond timestamp whose key ends in `_ms`
/// ("2026-10-19 14:03 UTC · 3 days ago"). Anything else has none.
pub fn property_time_hint(key: &str, value: &PropertyValue, now_ms: i64) -> Option<String> {
    let ms = property_time_ms(key, value)?;
    Some(match value {
        PropertyValue::Int(_) => format!(
            "{} UTC · {}",
            format_utc_minutes(ms),
            relative_age(ms, now_ms)
        ),
        _ => relative_age(ms, now_ms),
    })
}

/// The Unix milliseconds a date-like property names: an ISO 8601 date-time,
/// or a millisecond timestamp in 2000–2100 whose key ends in `_ms`.
pub fn property_time_ms(key: &str, value: &PropertyValue) -> Option<i64> {
    match value {
        PropertyValue::String(text) => parse_iso_datetime_ms(text),
        PropertyValue::Int(ms) if key.ends_with("_ms") && PLAUSIBLE_MS.contains(ms) => Some(*ms),
        _ => None,
    }
}

/// Unix milliseconds of an RFC 3339 date-time such as
/// `2026-10-21T22:03:00Z` or `2026-10-21T23:03:00.250+01:00`.
pub fn parse_iso_datetime_ms(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    if bytes.len() < 20 || !matches!(bytes[10], b'T' | b't' | b' ') {
        return None;
    }
    let digits = |range: std::ops::Range<usize>| -> Option<i64> {
        let part = text.get(range)?;
        part.bytes()
            .all(|byte| byte.is_ascii_digit())
            .then(|| part.parse().ok())
            .flatten()
    };
    if bytes[4] != b'-' || bytes[7] != b'-' || bytes[13] != b':' || bytes[16] != b':' {
        return None;
    }
    let (year, month, day) = (digits(0..4)?, digits(5..7)?, digits(8..10)?);
    let (hour, minute, second) = (digits(11..13)?, digits(14..16)?, digits(17..19)?);
    if !(1..=12).contains(&month)
        || !(1..=days_in_month(year, month)).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    let mut rest = &text[19..];
    let mut millis = 0;
    if let Some(fraction) = rest.strip_prefix('.') {
        let length = fraction.bytes().take_while(u8::is_ascii_digit).count();
        if length == 0 {
            return None;
        }
        let padded = format!("{:0<3}", &fraction[..length.min(3)]);
        millis = padded.parse::<i64>().ok()?;
        rest = &fraction[length..];
    }
    let offset_minutes = match rest {
        "Z" | "z" => 0,
        offset if offset.len() == 6 && offset.as_bytes()[3] == b':' => {
            let sign = match offset.as_bytes()[0] {
                b'+' => 1,
                b'-' => -1,
                _ => return None,
            };
            let hours: i64 = offset[1..3].parse().ok()?;
            let minutes: i64 = offset[4..6].parse().ok()?;
            sign * (hours * 60 + minutes)
        }
        _ => return None,
    };
    let seconds = days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second
        - offset_minutes * 60;
    Some(seconds * 1_000 + millis)
}

/// "just now", "5 minutes ago", "in 2 days", ... counted in whole units.
pub fn relative_age(then_ms: i64, now_ms: i64) -> String {
    let delta = now_ms - then_ms;
    let seconds = delta.unsigned_abs() / 1_000;
    if seconds < 60 {
        return "just now".into();
    }
    const UNITS: [(u64, &str); 6] = [
        (60, "minute"),
        (3_600, "hour"),
        (86_400, "day"),
        (604_800, "week"),
        (2_629_746, "month"),
        (31_556_952, "year"),
    ];
    let (size, name) = UNITS
        .iter()
        .rev()
        .find(|(size, _)| seconds >= *size)
        .copied()
        .unwrap_or(UNITS[0]);
    let count = seconds / size;
    let unit = if count == 1 {
        name.to_string()
    } else {
        format!("{name}s")
    };
    if delta >= 0 {
        format!("{count} {unit} ago")
    } else {
        format!("in {count} {unit}")
    }
}

/// `YYYY-MM-DD HH:MM` in UTC.
fn format_utc_minutes(ms: i64) -> String {
    let seconds = ms.div_euclid(1_000);
    let (days, of_day) = (seconds.div_euclid(86_400), seconds.rem_euclid(86_400));
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}",
        of_day / 3_600,
        of_day % 3_600 / 60
    )
}

fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap_year(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The inverse of [`days_from_civil`].
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let days = days + 719_468;
    let era = days.div_euclid(146_097);
    let day_of_era = days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    (year_of_era + era * 400 + i64::from(month <= 2), month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates_round_trip_through_day_counts() {
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(days_from_civil(2000, 2, 29)), (2000, 2, 29));
        for days in -800_000..800_000 {
            let (year, month, day) = civil_from_days(days);
            assert_eq!(days_from_civil(year, month, day), days);
        }
    }
}
