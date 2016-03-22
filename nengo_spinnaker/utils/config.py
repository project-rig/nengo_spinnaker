try:
    from nengo.exceptions import ConfigError
except ImportError:
    ConfigError = KeyError


def getconfig(config, object, name, default=None):
    """Get a configuration parameter that may or may not have been added to the
    config.
    """
    try:
        return getattr(config[object], name, default)
    except ConfigError:
        # This implies that the configuration hasn't been enabled.
        return default
