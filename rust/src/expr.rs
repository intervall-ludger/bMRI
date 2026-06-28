//! Tiny expression compiler for user-defined fit models.
//!
//! Supports the variable `x` (the independent axis) and the parameter names
//! `p0`, `p1`, ..., `p7` (positional access). For readability the parser
//! also accepts the common semantic aliases `S0`, `T`, `T2`, `T2s`, `T1`,
//! `T1rho`, `D`, `D_star`, `K`, `f`, `alpha`, `C`, `offset` which all map
//! to `p0..p7` in declaration order.
//!
//! Grammar (Pratt parser):
//!   expr   := term (('+'|'-') term)*
//!   term   := factor (('*'|'/') factor)*
//!   factor := atom ('^' factor)?
//!   atom   := number
//!          |  ident
//!          |  ident '(' expr (',' expr)* ')'
//!          |  '-' atom
//!          |  '(' expr ')'
//!
//! Functions supported: exp, log, sin, cos, tan, sqrt, abs, pow, min, max.
//!
//! The compiled program is a Vec<OpCode>. Evaluation walks the stack and
//! handles a single pixel/echo in roughly nanoseconds.

#[derive(Clone, Debug)]
enum OpCode {
    PushConst(f64),
    PushX,
    PushParam(usize),
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    CallExp,
    CallLog,
    CallSin,
    CallCos,
    CallTan,
    CallSqrt,
    CallAbs,
    CallMin,
    CallMax,
    CallPow,
}

pub struct Program {
    code: Vec<OpCode>,
}

impl Program {
    pub fn eval(&self, x: f64, p: &[f64]) -> f64 {
        // Most expressions stay shallow. A 16-slot stack covers any realistic
        // relaxometry model.
        let mut stack: [f64; 32] = [0.0; 32];
        let mut sp: usize = 0;
        for op in &self.code {
            match op {
                OpCode::PushConst(v) => {
                    stack[sp] = *v;
                    sp += 1;
                }
                OpCode::PushX => {
                    stack[sp] = x;
                    sp += 1;
                }
                OpCode::PushParam(idx) => {
                    stack[sp] = if *idx < p.len() { p[*idx] } else { 0.0 };
                    sp += 1;
                }
                OpCode::Neg => {
                    stack[sp - 1] = -stack[sp - 1];
                }
                OpCode::Add => {
                    sp -= 1;
                    stack[sp - 1] += stack[sp];
                }
                OpCode::Sub => {
                    sp -= 1;
                    stack[sp - 1] -= stack[sp];
                }
                OpCode::Mul => {
                    sp -= 1;
                    stack[sp - 1] *= stack[sp];
                }
                OpCode::Div => {
                    sp -= 1;
                    stack[sp - 1] /= stack[sp];
                }
                OpCode::Pow | OpCode::CallPow => {
                    sp -= 1;
                    stack[sp - 1] = stack[sp - 1].powf(stack[sp]);
                }
                OpCode::CallExp => {
                    stack[sp - 1] = stack[sp - 1].exp();
                }
                OpCode::CallLog => {
                    stack[sp - 1] = stack[sp - 1].ln();
                }
                OpCode::CallSin => {
                    stack[sp - 1] = stack[sp - 1].sin();
                }
                OpCode::CallCos => {
                    stack[sp - 1] = stack[sp - 1].cos();
                }
                OpCode::CallTan => {
                    stack[sp - 1] = stack[sp - 1].tan();
                }
                OpCode::CallSqrt => {
                    stack[sp - 1] = stack[sp - 1].sqrt();
                }
                OpCode::CallAbs => {
                    stack[sp - 1] = stack[sp - 1].abs();
                }
                OpCode::CallMin => {
                    sp -= 1;
                    stack[sp - 1] = stack[sp - 1].min(stack[sp]);
                }
                OpCode::CallMax => {
                    sp -= 1;
                    stack[sp - 1] = stack[sp - 1].max(stack[sp]);
                }
            }
        }
        stack[0]
    }
}

#[derive(Clone, Debug)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
}

fn tokenize(src: &str) -> Result<Vec<Token>, String> {
    let mut out = Vec::new();
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
        {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // optional exponent
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                i += 1;
                if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                    i += 1;
                }
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let s: String = chars[start..i].iter().collect();
            let v: f64 = s.parse().map_err(|e| format!("bad number '{s}': {e}"))?;
            out.push(Token::Number(v));
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let s: String = chars[start..i].iter().collect();
            out.push(Token::Ident(s));
            continue;
        }
        match c {
            '+' => out.push(Token::Plus),
            '-' => out.push(Token::Minus),
            '*' => {
                if i + 1 < chars.len() && chars[i + 1] == '*' {
                    out.push(Token::Caret);
                    i += 2;
                    continue;
                }
                out.push(Token::Star);
            }
            '/' => out.push(Token::Slash),
            '^' => out.push(Token::Caret),
            '(' => out.push(Token::LParen),
            ')' => out.push(Token::RParen),
            ',' => out.push(Token::Comma),
            other => return Err(format!("unexpected character '{other}'")),
        }
        i += 1;
    }
    Ok(out)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    n_params: usize,
}

fn alias_to_index(name: &str) -> Option<usize> {
    // Generic positional names p0..p7
    if name.starts_with('p') {
        if let Ok(idx) = name[1..].parse::<usize>() {
            return Some(idx);
        }
    }
    // Common semantic aliases — first one declared wins.
    None
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn eat(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect_paren_close(&mut self) -> Result<(), String> {
        match self.eat() {
            Some(Token::RParen) => Ok(()),
            other => Err(format!("expected ')', got {:?}", other)),
        }
    }

    fn parse_expr(&mut self, code: &mut Vec<OpCode>) -> Result<(), String> {
        self.parse_term(code)?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.eat();
                    self.parse_term(code)?;
                    code.push(OpCode::Add);
                }
                Some(Token::Minus) => {
                    self.eat();
                    self.parse_term(code)?;
                    code.push(OpCode::Sub);
                }
                _ => return Ok(()),
            }
        }
    }

    fn parse_term(&mut self, code: &mut Vec<OpCode>) -> Result<(), String> {
        self.parse_factor(code)?;
        loop {
            match self.peek() {
                Some(Token::Star) => {
                    self.eat();
                    self.parse_factor(code)?;
                    code.push(OpCode::Mul);
                }
                Some(Token::Slash) => {
                    self.eat();
                    self.parse_factor(code)?;
                    code.push(OpCode::Div);
                }
                _ => return Ok(()),
            }
        }
    }

    fn parse_factor(&mut self, code: &mut Vec<OpCode>) -> Result<(), String> {
        self.parse_atom(code)?;
        if matches!(self.peek(), Some(Token::Caret)) {
            self.eat();
            self.parse_factor(code)?; // right-associative
            code.push(OpCode::Pow);
        }
        Ok(())
    }

    fn parse_atom(&mut self, code: &mut Vec<OpCode>) -> Result<(), String> {
        match self.eat() {
            Some(Token::Number(v)) => {
                code.push(OpCode::PushConst(v));
                Ok(())
            }
            Some(Token::Minus) => {
                self.parse_atom(code)?;
                code.push(OpCode::Neg);
                Ok(())
            }
            Some(Token::LParen) => {
                self.parse_expr(code)?;
                self.expect_paren_close()
            }
            Some(Token::Ident(name)) => {
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.eat(); // consume '('
                    let mut nargs = 0;
                    if !matches!(self.peek(), Some(Token::RParen)) {
                        self.parse_expr(code)?;
                        nargs += 1;
                        while matches!(self.peek(), Some(Token::Comma)) {
                            self.eat();
                            self.parse_expr(code)?;
                            nargs += 1;
                        }
                    }
                    self.expect_paren_close()?;
                    match (name.as_str(), nargs) {
                        ("exp", 1) => code.push(OpCode::CallExp),
                        ("log" | "ln", 1) => code.push(OpCode::CallLog),
                        ("sin", 1) => code.push(OpCode::CallSin),
                        ("cos", 1) => code.push(OpCode::CallCos),
                        ("tan", 1) => code.push(OpCode::CallTan),
                        ("sqrt", 1) => code.push(OpCode::CallSqrt),
                        ("abs", 1) => code.push(OpCode::CallAbs),
                        ("min", 2) => code.push(OpCode::CallMin),
                        ("max", 2) => code.push(OpCode::CallMax),
                        ("pow", 2) => code.push(OpCode::CallPow),
                        (other, n) => {
                            return Err(format!("unknown function '{other}' with {n} args"));
                        }
                    }
                    Ok(())
                } else if name == "x" {
                    code.push(OpCode::PushX);
                    Ok(())
                } else if let Some(idx) = parse_param_name(&name, self.n_params) {
                    code.push(OpCode::PushParam(idx));
                    Ok(())
                } else {
                    Err(format!("unknown identifier '{name}'"))
                }
            }
            other => Err(format!("unexpected token {:?}", other)),
        }
    }
}

/// Recognise p0..pN-1 plus a small set of semantic aliases. Semantic aliases
/// map deterministically to p0..p3:
///   S0 -> p0   T,T1,T2,T2s,T1rho,D -> p1   K,D_star,T_long -> p2
///   offset,C,f,alpha -> p3 (for 4-param models)
///
/// Users who don't want guesswork can always write p0..pN-1 directly.
fn parse_param_name(name: &str, n_params: usize) -> Option<usize> {
    if let Some(stripped) = name.strip_prefix('p') {
        if let Ok(i) = stripped.parse::<usize>() {
            return if i < n_params { Some(i) } else { None };
        }
    }
    if let Some(idx) = alias_to_index(name) {
        return Some(idx);
    }
    let idx = match name {
        "S0" => 0,
        "T" | "T2" | "T2s" | "T2star" | "T1" | "T1rho" | "D" | "ADC" => 1,
        "K" | "D_star" | "Dstar" | "T_long" | "Tl" => 2,
        "offset" | "C" | "f" | "alpha" => {
            if n_params >= 4 {
                3
            } else if n_params == 3 {
                2
            } else {
                return None;
            }
        }
        _ => return None,
    };
    if idx < n_params {
        Some(idx)
    } else {
        None
    }
}

pub fn compile(src: &str, n_params: usize) -> Result<Program, String> {
    let tokens = tokenize(src)?;
    let mut parser = Parser {
        tokens,
        pos: 0,
        n_params,
    };
    let mut code = Vec::new();
    parser.parse_expr(&mut code)?;
    if parser.pos != parser.tokens.len() {
        return Err(format!(
            "unconsumed tokens at position {} ('{:?}')",
            parser.pos,
            parser.tokens.get(parser.pos)
        ));
    }
    Ok(Program { code })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_mono_exp() {
        // S0 * exp(-x/T) + C, with p = [1, 50, 0]
        let prog = compile("S0 * exp(-x/T) + C", 3).unwrap();
        let v = prog.eval(50.0, &[1.0, 50.0, 0.0]);
        assert!((v - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn evaluates_with_pn_names() {
        let prog = compile("p0 * exp(-x * p1)", 2).unwrap();
        let v = prog.eval(2.0, &[3.0, 0.5]);
        assert!((v - 3.0 * (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn pow_operator() {
        let prog = compile("x^2 + 1", 1).unwrap();
        assert!((prog.eval(3.0, &[0.0]) - 10.0).abs() < 1e-12);
    }
}
