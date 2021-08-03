from dglke.dist_main import dist_main
import sys
import re
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(dist_main())
