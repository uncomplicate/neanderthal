package uncomplicate.neanderthal.protocols;

import java.nio.ByteBuffer;

public interface RealBufferAccessor {

    double get (ByteBuffer buf, long index);

    void set (ByteBuffer buf, long index, double value);

}
