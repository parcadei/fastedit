package main

const Limit = 10

func Alpha(x int) int {
	return x + 1
}

func Gamma(z int) int {
	return z - 1
}

type Item struct {
	Name string
}

func Beta(y int) int {
	return y * 2
}
