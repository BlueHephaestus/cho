import sys, time
for i in range(100):
    time.sleep(1)
    sys.stdout.write("\r%i" % i)
    sys.stdout.flush()
