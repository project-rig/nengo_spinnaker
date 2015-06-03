import mock
import pytest
from rig.machine import Cores, SDRAM
import six
import struct
import tempfile

from nengo_spinnaker.builder.builder import (
    Model, Signal, ObjectPort, OutputPort
)
from nengo_spinnaker.operators import SDPReceiver
from nengo_spinnaker.operators.sdp_receiver import SystemRegion


class TestSDPReceiver(object):
    def test_make_vertices(self):
        """Check that the SDPReceiver make_vertices method looks at the signals
        and connections to make one sdp-rx vertex per outgoing connection and
        stores these as dictionaries.
        """
        # Create the SDPReceiver, check that it currently has no connections
        # mapped to vertices.
        sdp_rx = SDPReceiver()
        assert sdp_rx.connection_vertices == dict()

        # Create a model containing some signals and connections originating
        # from sdp_rx, when calling make_vertices check that one vertex is
        # created per connection and that the dictionary is updated
        # accordingly.
        model = Model()

        conn_a = mock.Mock(name="connection a")
        conn_a.size_out = 1
        conn_a.post_slice = slice(0, 1)
        conn_a.post_obj.size_in = 3
        ks_a = model.keyspaces["nengo"](object=0, connection=0)
        sig_a = Signal(ObjectPort(sdp_rx, OutputPort.standard), None, ks_a)

        conn_b = mock.Mock(name="connection b")
        conn_b.size_out = 2
        conn_b.post_slice = slice(None)
        conn_b.post_obj.size_in = 2
        ks_b = model.keyspaces["nengo"](object=0, connection=1)
        sig_b = Signal(ObjectPort(sdp_rx, OutputPort.standard), None, ks_b)

        model.connections_signals = {
            conn_a: sig_a,
            conn_b: sig_b,
        }

        # Make the vertices
        nls = sdp_rx.make_vertices(model, 1)  # TODO Remove number of steps
        assert len(nls.vertices) == 2
        assert nls.load_function == sdp_rx.load_to_machine

        for conn, vx in six.iteritems(sdp_rx.connection_vertices):
            print(conn, vx)
            assert conn is conn_a or conn is conn_b
            assert vx in nls.vertices
            assert vx.resources[Cores] == 1
            assert vx.resources[SDRAM] == 3*4 + 2*4 + 4*conn.size_out

            assert sdp_rx._sys_regions[vx].size_out == conn.size_out
            assert vx in sdp_rx._key_regions

    def test_make_vertices_none_required(self):
        sdp_rx = SDPReceiver()

        # No outgoing connections
        model = Model()

        # Make the vertices
        nls = sdp_rx.make_vertices(model, 1)  # TODO Remove number of steps
        assert len(nls.vertices) == 0

        assert sdp_rx.connection_vertices == dict()


class TestSystemRegion(object):
    @pytest.mark.parametrize("machine_timestep, size_out",
                             [(1000, 3), (2000, 1)])
    def test_all(self, machine_timestep, size_out):
        # Create a system region
        region = SystemRegion(machine_timestep, size_out)

        # Check that the size is correct
        assert region.sizeof(slice(None)) == 8

        # Write the data to a file and check that it is correct
        fp = tempfile.TemporaryFile()
        region.write_region_to_file(fp)

        fp.seek(0)
        assert fp.read() == struct.pack("<2I", machine_timestep, size_out)
