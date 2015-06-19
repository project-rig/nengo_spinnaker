import nengo
import numpy as np
import threading

from nengo_spinnaker.builder.builder import (
    InputPort, ObjectPort, OutputPort, spec)
from nengo_spinnaker.operators import Filter, ValueSink, ValueSource
from nengo_spinnaker.utils.config import getconfig


class NodeIOController(object):
    """Object which controls communication between SpiNNaker and the host
    computer for transmitting output and receiving input for Node objects.

    `NodeIOController` itself should not be used, instead use one of its
    subclasses as these provide specific IO implementations. Subclasses are
    expected to implement the following methods to allow building of the model:

        - `get_spinnaker_source_for_node(model, connection)`
        - `get_spinnaker_sink_for_node(model, connection)`

    These methods should perform the same tasks as performed by
    :py:attr:`~nengo_spinnaker.builder.Model.source_getters` and
    :py:attr:`~nengo_spinnaker.builder.Model.sink_getters` respectively.

    A `NodeIOController` can be used to build a model::

        io = NodeIOController()  # Should be a subclass
        model.build(network,
                    extra_builders={Node: io.build_node},
                    extra_source_getters={Node: io.get_node_source},
                    extra_sink_getters={Node: io.get_node_sink},
                    extra_probe_builders={None: io.build_probe}
                    )

    If the IO controller is the sole supplier of extra builders, etc., then the
    :py:attr:`~.builder_kwargs` attribute may be used::

        model.build(network, **io.builder_kwargs)

    Once the model is built the `NodeIOController` will contain the parts of
    the model which should be simulated on the host, and can be used to perform
    communication between SpiNNaker and the host. Subclasses should also
    implement :py:meth:`~.set_node_output` for setting Node values and should
    write received node inputs into :py:attr:`~.node_inputs`, a dictionary with
    Nodes as the keys and Numpy arrays as the values using the
    :py:attr:`~.node_inputs_lock`.  Subclasses should override
    :py:meth:`~.prepare` if they need access to a netlist.

    Finally, subclasses should implement a `spawn` method, which returns a
    thread which manages IO.  This thread must have a `stop` method which
    causes the thread to stop executing. See the Ethernet implementation for an
    example.
    """

    def __init__(self):
        # Create a network which will contain everything that is to be
        # simulated on the host computer.
        self.host_network = nengo.Network()

        # Store objects that we've added
        self._f_of_t_nodes = dict()
        self._passthrough_nodes = dict()
        self._added_nodes = set()
        self._added_conns = set()
        self._input_nodes = dict()
        self._output_nodes = dict()

        # Cached node inputs
        self.node_input_lock = threading.Lock()
        self.node_input = dict()

    @property
    def builder_kwargs(self):
        """Keyword arguments that can be used with the standard model builder.
        """
        return {
            "extra_builders": {nengo.Node: self.build_node},
            "extra_source_getters": {nengo.Node: self.get_node_source},
            "extra_sink_getters": {nengo.Node: self.get_node_sink},
            "extra_probe_builders": {nengo.Node: self.build_node_probe},
        }

    def _add_node(self, node):
        """Add a Node to the host network."""
        if node not in self._added_nodes:
            self.host_network.add(node)
            self._added_nodes.add(node)

    def _add_connection(self, connection):
        """Add a connection to the host network."""
        if connection not in self._added_conns:
            self.host_network.add(connection)
            self._added_conns.add(connection)

    def build_node(self, model, node):
        """Modify the model to build the Node."""
        f_of_t = node.size_in == 0 and (
            not callable(node.output) or
            getconfig(model.config, node, "function_of_time", False)
        )

        if node.output is None:
            # If the Node is a passthrough Node then create a new filter object
            # for it.
            op = Filter(node.size_in)
            self._passthrough_nodes[node] = op
            model.object_operators[node] = op
        elif f_of_t:
            # If the Node is a function of time then add a new value source for
            # it.
            vs = ValueSource(
                node.output,
                node.size_out,
                getconfig(model.config, node, "function_of_time_period")
            )
            self._f_of_t_nodes[node] = vs
            model.object_operators[node] = vs
        else:
            with self.host_network:
                self._add_node(node)

    def build_node_probe(self, model, probe):
        """Modify the model to build the Probe."""
        # Create a new ValueSink for the probe and add this to the model.
        model.object_operators[probe] = ValueSink(probe, model.dt)

        # Create a new connection from the Node to the Probe and then get the
        # model to build this.
        seed = model.seeds[probe]
        conn = nengo.Connection(probe.target, probe, synapse=probe.synapse,
                                seed=seed, add_to_container=False)
        model.make_connection(conn)

    def get_node_source(self, model, cn):
        """Get the source for a connection originating from a Node."""
        if cn.pre_obj in self._passthrough_nodes:
            # If the Node is a passthrough Node then we return a reference
            # to the Filter operator we created earlier regardless.
            return spec(ObjectPort(self._passthrough_nodes[cn.pre_obj],
                                   OutputPort.standard))
        elif cn.pre_obj in self._f_of_t_nodes:
            # If the Node is a function of time Node then we return a
            # reference to the value source we created earlier.
            return spec(ObjectPort(self._f_of_t_nodes[cn.pre_obj],
                                   OutputPort.standard))
        elif (type(cn.post_obj) is nengo.Node and
                cn.post_obj not in self._passthrough_nodes):
            # If this connection goes from a Node to another Node (exactly, not
            # any subclasses) then we just add both nodes and the connection to
            # the host model.
            with self.host_network:
                self._add_node(cn.pre_obj)
                self._add_node(cn.post_obj)
                self._add_connection(cn)

            # Return None to indicate that the connection should not be
            # represented by a signal on SpiNNaker.
            return None
        else:
            # Otherwise, we create a new OutputNode for the Node at the
            # start of the given connection, then add both it and the Node
            # to the host network, with a joining connection.
            with self.host_network:
                # Create the output Node if necessary
                if cn.pre_obj not in self._output_nodes:
                    self._add_node(cn.pre_obj)
                    self._output_nodes[cn.pre_obj] = \
                        OutputNode(cn.pre_obj, self)

                    output_node = self._output_nodes[cn.pre_obj]
                    nengo.Connection(cn.pre_obj, output_node, synapse=None)

            # Return a specification that describes how the signal should
            # be represented on SpiNNaker.
            return self.get_spinnaker_source_for_node(model, cn)

    def get_spinnaker_source_for_node(self, model, cn):  # pragma: no cover
        """Get the source for a connection originating from a Node.

        **OVERRIDE THIS METHOD** when creating a new IO controller. The return
        type is as documented for
        :py:attr:`~nengo_spinnaker.builder.Model.source_getters`.
        """
        raise NotImplementedError

    def get_node_sink(self, model, cn):
        """Get the sink for a connection terminating at a Node."""
        if cn.post_obj in self._passthrough_nodes:
            # If the Node is a passthrough Node then we return a reference
            # to the Filter operator we created earlier regardless.
            return spec(ObjectPort(self._passthrough_nodes[cn.post_obj],
                                   InputPort.standard))
        elif (type(cn.pre_obj) is nengo.Node and
                cn.pre_obj not in self._passthrough_nodes):
            # If this connection goes from a Node to another Node (exactly, not
            # any subclasses) then we just add both nodes and the connection to
            # the host model.
            with self.host_network:
                self._add_node(cn.pre_obj)
                self._add_node(cn.post_obj)
                self._add_connection(cn)

            # Return None to indicate that the connection should not be
            # represented by a signal on SpiNNaker.
            return None
        else:
            # Otherwise we create a new InputNode for the Node at the end
            # of the given connection, then add both it and the Node to the
            # host network with a joining connection.
            with self.host_network:
                self._add_node(cn.post_obj)

                # Create the input node AND connection if necessary
                if cn.post_obj not in self._input_nodes:
                    self._input_nodes[cn.post_obj] = \
                        InputNode(cn.post_obj, self)

                    input_node = self._input_nodes[cn.post_obj]
                    nengo.Connection(input_node, cn.post_obj, synapse=None)

            # Return a specification that describes how the signal should
            # be represented on SpiNNaker.
            return self.get_spinnaker_sink_for_node(model, cn)

    def get_spinnaker_sink_for_node(self, model, cn):  # pragma: no cover
        """Get the sink for a connection terminating at a Node.

        **OVERRIDE THIS METHOD** when creating a new IO controller. The return
        type is as documented for
        :py:attr:`~nengo_spinnaker.builder.Model.sink_getters`.
        """
        raise NotImplementedError

    def prepare(self, model, controller, netlist):
        """Prepare the Node controller to work with the given model, netlist
        and machine controller.
        """
        pass

    def set_node_output(self, node, value):  # pragma: no cover
        """Transmit the value output by a Node.

        **OVERRIDE THIS METHOD** when creating a new IO controller.
        """
        raise NotImplementedError

    def spawn(self):  # pragma: no cover
        """Get a new thread which will handle IO for a period of simulation
        time.

        The returned thread _must_ implement a method called `stop` which
        terminates the execution of the thread.

        **OVERRIDE THIS METHOD** when creating a new IO controller.
        """
        raise NotImplementedError

    def close(self):
        """Close the NodeIOController."""
        pass


class InputNode(nengo.Node):
    """Node which queries the IO controller for the input to a Node from."""
    def __init__(self, node, controller):
        self.size_in = 0
        self.size_out = node.size_in
        self.target = node
        self.controller = controller
        controller.node_input[node] = np.zeros(self.size_out)

    def output(self, t):
        """This should ask the controller for the input value for the target
        Node.
        """
        with self.controller.node_input_lock:
            return self.controller.node_input[self.target]


class OutputNode(nengo.Node):
    """Node which can accept output from another Node and transmit it to a
    SpiNNaker machine via the IO controller.
    """
    def __init__(self, node, controller):
        self.size_in = node.size_out
        self.size_out = 0
        self.target = node
        self.controller = controller

    def output(self, t, value):
        """This should inform the controller of the output value of the
        target.
        """
        self.controller.set_node_output(self.target, value)
