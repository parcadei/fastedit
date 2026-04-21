"""Tests for the AST analyzer across multiple languages."""

import pytest
from fastedit.data_gen.ast_analyzer import (
    analyze_file,
    detect_language,
    validate_parse,
    get_node_at_lines,
    parse_code,
    count_ast_nodes,
)

# -- Language detection --

def test_detect_python():
    assert detect_language("app.py") == "python"

def test_detect_typescript():
    assert detect_language("index.ts") == "typescript"
    assert detect_language("App.tsx") == "tsx"

def test_detect_rust():
    assert detect_language("main.rs") == "rust"

def test_detect_go():
    assert detect_language("main.go") == "go"

def test_detect_unknown():
    assert detect_language("README.md") is None


# -- Python analysis --

PYTHON_CODE = '''
import os
from pathlib import Path

class FileProcessor:
    """Process files."""

    def __init__(self, root: Path):
        self.root = root
        self.files = []

    def scan(self) -> list[Path]:
        """Scan for files."""
        for entry in os.scandir(self.root):
            if entry.is_file():
                self.files.append(Path(entry.path))
        return self.files

    def filter_by_ext(self, ext: str) -> list[Path]:
        return [f for f in self.files if f.suffix == ext]

def count_lines(path: Path) -> int:
    with open(path) as f:
        return sum(1 for _ in f)

def main():
    proc = FileProcessor(Path("."))
    files = proc.scan()
    py_files = proc.filter_by_ext(".py")
    for f in py_files:
        print(f"{f}: {count_lines(f)} lines")
'''

def test_python_analysis():
    structure = analyze_file(PYTHON_CODE, "python", "test.py")
    assert structure.language == "python"
    assert not structure.has_parse_errors
    assert len(structure.classes) == 1
    assert structure.classes[0].name == "FileProcessor"
    # Functions: __init__, scan, filter_by_ext (inside class) + count_lines, main (top-level)
    assert len(structure.functions) >= 2  # at least count_lines and main
    assert len(structure.imports) == 2
    assert structure.complexity_bucket in ("simple", "moderate", "complex")

def test_python_parse_valid():
    assert validate_parse(PYTHON_CODE, "python")
    assert not validate_parse("def foo(:\n  pass", "python")


# -- TypeScript analysis --

TYPESCRIPT_CODE = '''
import { useState, useEffect } from 'react';
import type { User } from './types';

interface UserListProps {
  initialUsers: User[];
  onSelect: (user: User) => void;
}

export function UserList({ initialUsers, onSelect }: UserListProps) {
  const [users, setUsers] = useState<User[]>(initialUsers);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchUsers().then(setUsers);
  }, []);

  async function fetchUsers(): Promise<User[]> {
    setLoading(true);
    const response = await fetch('/api/users');
    setLoading(false);
    return response.json();
  }

  const handleClick = (user: User) => {
    onSelect(user);
  };

  if (loading) return <div>Loading...</div>;

  return (
    <ul>
      {users.map(user => (
        <li key={user.id} onClick={() => handleClick(user)}>
          {user.name}
        </li>
      ))}
    </ul>
  );
}
'''

def test_typescript_analysis():
    structure = analyze_file(TYPESCRIPT_CODE, "tsx", "UserList.tsx")
    assert structure.language == "tsx"
    assert not structure.has_parse_errors
    assert len(structure.functions) >= 1  # UserList at minimum
    assert len(structure.imports) >= 1


# -- Rust analysis --

RUST_CODE = '''
use std::collections::HashMap;
use std::io::{self, Read};

pub struct Config {
    settings: HashMap<String, String>,
}

impl Config {
    pub fn new() -> Self {
        Config {
            settings: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&String> {
        self.settings.get(key)
    }

    pub fn set(&mut self, key: String, value: String) {
        self.settings.insert(key, value);
    }
}

fn read_config_file(path: &str) -> io::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

pub fn load_config(path: &str) -> io::Result<Config> {
    let contents = read_config_file(path)?;
    let mut config = Config::new();
    for line in contents.lines() {
        if let Some((key, value)) = line.split_once('=') {
            config.set(key.trim().to_string(), value.trim().to_string());
        }
    }
    Ok(config)
}
'''

def test_rust_analysis():
    structure = analyze_file(RUST_CODE, "rust", "config.rs")
    assert structure.language == "rust"
    assert not structure.has_parse_errors
    assert len(structure.functions) >= 2  # read_config_file, load_config + impl methods
    assert len(structure.classes) >= 1  # struct Config or impl Config


# -- Go analysis --

GO_CODE = '''
package main

import (
	"fmt"
	"os"
	"strings"
)

type Server struct {
	Host string
	Port int
}

func NewServer(host string, port int) *Server {
	return &Server{Host: host, Port: port}
}

func (s *Server) Address() string {
	return fmt.Sprintf("%s:%d", s.Host, s.Port)
}

func (s *Server) Start() error {
	addr := s.Address()
	fmt.Printf("Starting server on %s\\n", addr)
	return nil
}

func main() {
	host := os.Getenv("HOST")
	if host == "" {
		host = "localhost"
	}
	srv := NewServer(host, 8080)
	if err := srv.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\\n", err)
		os.Exit(1)
	}
}
'''

def test_go_analysis():
    structure = analyze_file(GO_CODE, "go", "main.go")
    assert structure.language == "go"
    assert not structure.has_parse_errors
    assert len(structure.functions) >= 3  # NewServer, main + methods


# -- Java analysis --

JAVA_CODE = '''
package com.example;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

public class UserService {
    private final List<User> users;

    public UserService() {
        this.users = new ArrayList<>();
    }

    public void addUser(User user) {
        users.add(user);
    }

    public Optional<User> findById(int id) {
        return users.stream()
            .filter(u -> u.getId() == id)
            .findFirst();
    }

    public List<User> findAll() {
        return List.copyOf(users);
    }
}
'''

def test_java_analysis():
    structure = analyze_file(JAVA_CODE, "java", "UserService.java")
    assert structure.language == "java"
    assert not structure.has_parse_errors
    assert len(structure.classes) >= 1
    assert len(structure.imports) >= 1


# -- JavaScript analysis --

JS_CODE = '''
const express = require('express');
const { validateInput } = require('./middleware');

function createApp(config) {
  const app = express();

  app.use(express.json());

  app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  app.post('/api/data', validateInput, (req, res) => {
    const result = processData(req.body);
    res.json(result);
  });

  return app;
}

function processData(data) {
  return {
    processed: true,
    count: Object.keys(data).length,
    timestamp: Date.now(),
  };
}

module.exports = { createApp, processData };
'''

def test_javascript_analysis():
    structure = analyze_file(JS_CODE, "javascript", "app.js")
    assert structure.language == "javascript"
    assert not structure.has_parse_errors
    assert len(structure.functions) >= 2  # createApp, processData


# -- Cross-cutting tests --

def test_parse_validity_check():
    assert validate_parse("def foo():\n  return 1", "python")
    assert validate_parse("function foo() { return 1; }", "javascript")
    assert validate_parse("fn foo() -> i32 { 1 }", "rust")
    assert validate_parse("func foo() int { return 1 }", "go")

def test_count_ast_nodes():
    tree = parse_code("def foo():\n  return 1", "python")
    count = count_ast_nodes(tree)
    assert count > 0

def test_node_at_lines():
    nodes = get_node_at_lines(PYTHON_CODE, "python", 25, 28)
    assert len(nodes) > 0  # Should find the count_lines function

def test_file_structure_complexity():
    simple = analyze_file("def foo():\n  return 1", "python")
    assert simple.complexity_bucket == "simple"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
