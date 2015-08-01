package uncomplicate.neanderthal.protocols;

public interface Block {

    Object entryType ();

    Object buffer ();

    long stride ();

    long order ();

    long count ();

}
