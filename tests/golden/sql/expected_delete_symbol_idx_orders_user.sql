CREATE TABLE users (
  id INT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT
);

