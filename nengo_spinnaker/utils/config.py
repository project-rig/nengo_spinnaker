def getconfig(config, object, name, default=None):
    """Get a configuration parameter that may or may not have been added to the
    config.
    """
    try:
        return getattr(config[object], name, default)
    except KeyError:
        # This implies that the configuration hasn't been enabled.
        return default
