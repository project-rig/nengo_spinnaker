import pkg_resources


def get_application(app_name):
    app_name = "binaries/nengo_{}.aplx".format(app_name)
    return pkg_resources.resource_filename("nengo_spinnaker", app_name)
