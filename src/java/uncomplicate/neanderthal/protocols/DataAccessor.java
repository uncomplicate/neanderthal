package uncomplicate.neanderthal.protocols;

import java.nio.ByteBuffer;

public interface DataAccessor {

    Object entryType ();

    long entryWidth ();

    long count (Object buf);

    Object createDataSource (long n);
}
