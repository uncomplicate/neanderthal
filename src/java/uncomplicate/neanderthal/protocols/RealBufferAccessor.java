package uncomplicate.neanderthal.protocols;

import java.nio.ByteBuffer;

public interface RealBufferAccessor {

    double get (ByteBuffer buf, long index);

    void set (ByteBuffer buf, long index, double value);

    Object toSeq (ByteBuffer buf, long stride);

    ByteBuffer directBuffer (long n);

    ByteBuffer slice (ByteBuffer buf, long k, long l);

}
