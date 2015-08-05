package uncomplicate.neanderthal.protocols;

import java.nio.ByteBuffer;

public interface BufferAccessor extends DataAccessor {

    ByteBuffer toBuffer (Object source);

    Object toSeq (ByteBuffer buf, long stride);

    ByteBuffer directBuffer (long n);

    ByteBuffer slice (ByteBuffer buf, long k, long l);

    long count (ByteBuffer buf);
}
