import multiprocessing as mp
import time

def job(a, q):
    l = 0
    for i in range(a):
        # l.append(i + i**2 + i**3)
        l += i + i ** 2 + i ** 3
    q.put(l)

def job1(a):
    l = 0
    for i in range(a):
        # l.append(i + i**2 + i**3)
        l += i
    return l



def job2(q):
    res = 0
    for i in range(1000000):
        res += i + i**2 + i**3
    q.put(res)


def multicore():
    q = mp.Queue()
    p1 = mp.Process(target=job2, args=(q,))
    p2 = mp.Process(target=job2, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()


if __name__ == '__main__':
    # s = time.time()
    # q = mp.Queue()
    # p1 = mp.Process(target=job, args=(1000000, q))
    # p1.start()
    # p1.join()
    # res1 = q.get()
    #
    # # job(10000)
    # print(time.time()-s)
    #
    # s = time.time()
    # job1(1000000)
    # print(time.time() - s)

    pool = mp.Pool(processes=mp.cpu_count())
    result = pool.map(job1, [3] * 8)
    print(result)