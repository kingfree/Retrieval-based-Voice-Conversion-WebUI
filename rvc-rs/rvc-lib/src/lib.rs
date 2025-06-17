/// Core functionality placeholder
pub fn greet(name: &str) -> String {
    format!("Hello, {name} from rvc-lib!")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        assert_eq!(greet("World"), "Hello, World from rvc-lib!");
    }
}
