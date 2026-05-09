# KRISHNA KUMAR TRIVEDI
# TE-B-25
# McCulloch Pitts ANDNOT Function

def andnot(x1, x2):
    w1 = 1
    w2 = -1
    theta = 1   # threshold

    yin = (w1 * x1) + (w2 * x2)

    if yin >= theta:
        return 1
    else:
        return 0


# All input combinations
inputs = [(1,1), (1,0), (0,1), (0,0)]

# Table Header
print("+----+----+----+")
print("| X1 | X2 | Y  |")
print("+----+----+----+")


for x1, x2 in inputs:
    output = andnot(x1, x2)
    print(f"| {x1}  | {x2}  | {output}  |")

print("+----+----+----+")
