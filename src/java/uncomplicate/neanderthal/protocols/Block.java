package uncomplicate.neanderthal.protocols;

public interface Block {

    Object entryType ();

    Object buffer ();

    long offset ();

    long stride ();

    long order ();

    long count ();

}
