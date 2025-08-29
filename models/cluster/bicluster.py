X = [
    [5,5,0,0],
    [5,4,0,0],
    [0,0,3,3],
    [0,0,4,4]
]

rows_top = [i for i,row in enumerate(X) if sum(row[:2]) > 0]
rows_bottom = [i for i,row in enumerate(X) if sum(row[2:]) > 0]
cols_left = [j for j in range(2)]
cols_right = [j for j in range(2,4)]

print("Row clusters:", [rows_top, rows_bottom])
print("Column clusters:", [cols_left, cols_right])
