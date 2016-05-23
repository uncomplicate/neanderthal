package uncomplicate.neanderthal.protocols;

import java.nio.ByteBuffer;

public interface DataAccessor {

    Object entryType ();

    long entryWidth ();

    long count (Object data);

    Object createDataSource (long n);

    Object initialize (Object data);

    Object initialize (Object data, Object value);

    Object wrapPrim (double data);

}
