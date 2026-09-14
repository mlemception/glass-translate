# Test fixtures

Greyscale crops of the pages in `more_comparisons/`, cut for the typesetting
regression tests.  Each is the smallest rectangle that still carries the
geometry under test; the OCR boxes that go with them are hard-coded in the
test modules (taken from `demo/cache/<stem>/segments_all.pkl`), so no test
needs an OCR engine.

| file | page | crop rectangle (x, y, w, h) | what it carries |
|---|---|---|---|
| `1ja_tailed_balloon.png` | `more_comparisons/1ja.jpg` | 585, 810, 135, 160 | one balloon with a tail at the bottom, two source columns (block 12 of the page) |
| `2ja_joined_balloons.png` | `more_comparisons/2ja.jpg` | 470, 780, 220, 175 | two balloons whose interiors are joined, four columns in the right one and one in the left (blocks 2, 11 and 12 of the page) |
| `4ja_joined_balloons.png` | `more_comparisons/4ja.jpg` | 500, 725, 230, 160 | three balloons sharing one patch of paper (blocks 9, 10 and 11 of the page) |
