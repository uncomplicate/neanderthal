package uncomplicate.neanderthal.protocols;

import java.nio.ByteBuffer;

public interface BufferAccessor extends DataAccessor {

    Object toSeq (ByteBuffer buf, long stride);

    ByteBuffer slice (ByteBuffer buf, long k, long l);

}
