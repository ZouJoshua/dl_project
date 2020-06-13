#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2020/6/13 11:36 上午
@File    : write_data_to_redis.py
@Desc    : 写redis

"""

# pip install redis-py-cluster
from rediscluster import RedisCluster




# 测试要写的数据填在这里
key = "XXX"
value = "XXX"

def write_redis(key, value):
    redis_nodes = [
        {'host': '127.0.0.1', 'port': 7000},
        {'host': '127.0.0.1', 'port': 7001},
        {'host': '127.0.0.1', 'port': 7002},
        {'host': '127.0.0.1', 'port': 7004},
        {'host': '127.0.0.1', 'port': 7005},
    ]
    try:
        redisconn = RedisCluster(startup_nodes=redis_nodes, decode_responses=True)
        redisconn.set(key, value)
        print(redisconn.get(key))
    except:
        print('error')

write_redis(key, value)