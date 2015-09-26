package uncomplicate.neanderthal.protocols;

public interface Block {

    Object entryType ();

    Object buffer ();

    Object offset ();

    long stride ();

    long order ();

    long count ();

}
