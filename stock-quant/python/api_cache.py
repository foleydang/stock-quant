"""
API缓存模块 - 减少重复查询
"""
import time
from functools import wraps

class APICache:
    """简单的内存缓存"""
    
    _cache = {}
    _ttl = {}
    
    @classmethod
    def get(cls, key, ttl=60):
        """获取缓存"""
        if key in cls._cache:
            if time.time() - cls._ttl.get(key, 0) < ttl:
                return cls._cache[key]
        return None
    
    @classmethod
    def set(cls, key, value, ttl=60):
        """设置缓存"""
        cls._cache[key] = value
        cls._ttl[key] = time.time()
    
    @classmethod
    def clear(cls):
        """清空缓存"""
        cls._cache.clear()
        cls._ttl.clear()

def cached(ttl=60):
    """缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            cached_val = APICache.get(key, ttl)
            if cached_val is not None:
                return cached_val
            result = func(*args, **kwargs)
            APICache.set(key, result, ttl)
            return result
        return wrapper
    return decorator
