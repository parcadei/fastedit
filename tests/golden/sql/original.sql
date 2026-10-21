CREATE TABLE users (
  id INT PRIMARY KEY,
  name TEXT NOT NULL
);

CREATE TABLE orders (
  id INT PRIMARY KEY,
  user_id INT
);

CREATE INDEX idx_orders_user ON orders (user_id);
