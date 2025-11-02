__all__ = ("GraspNetBaseLine",)


def __getattr__(name):
    if name == "GraspNetBaseLine":
        from .graspnet_baseline import GraspNetBaseLine

        return GraspNetBaseLine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
