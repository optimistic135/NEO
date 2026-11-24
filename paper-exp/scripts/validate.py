import sys
from run import main

if __name__ == "__main__":
    sys.argv.append("--validate")
    raise SystemExit(main())
