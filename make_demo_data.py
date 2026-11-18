"""Generate an original synthetic grammar corpus (no downloaded/private text)."""
from pathlib import Path

def make_text():
    return "".join(
        f"the {['small','bright','quiet','blue'][i%4]} "
        f"{['cat','bird','river','moon'][i%4]} is "
        f"{['near','above','below'][i%3]} the "
        f"{['tree','house','hill','lake','cloud'][i%5]}.\n"
        for i in range(300)
    )

if __name__ == "__main__":
    Path("examples").mkdir(exist_ok=True)
    Path("examples/synthetic.txt").write_text(make_text(),encoding="utf8")
