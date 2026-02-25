import re

with open("JITS/JITS_2_0220.tex", "r") as f:
    content = f.read()

start = content.find(r"\section{Related Work}")
end = content.find(r"\section{Materials and Methods}")
print(content[start:end])
