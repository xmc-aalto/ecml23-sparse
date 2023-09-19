import sys
import tensorflow as tf

if sys.argv[1] == "includes":
    flags = tf.sysconfig.get_compile_flags()
    include_dirs = []
    for flag in flags:
        if flag.startswith("-I"):
            include_dirs.append(flag[2:])
    print(";".join(include_dirs))
elif sys.argv[1] == "defines":
    flags = tf.sysconfig.get_compile_flags()
    defines = []
    for flag in flags:
        if flag.startswith("-D"):
            defines.append(flag[2:])
    print(";".join(defines))
elif sys.argv[1] == "other-compile":
    flags = tf.sysconfig.get_compile_flags()
    others = []
    for flag in flags:
        if not flag.startswith("-D") and not flag.startswith("-I"):
            others.append(flag)
    print(";".join(others))
elif sys.argv[1] == "link-dirs":
    flags = tf.sysconfig.get_link_flags()
    link_dirs = []
    for flag in flags:
        if flag.startswith("-L"):
            link_dirs.append(flag[2:])
    print(";".join(link_dirs))
elif sys.argv[1] == "link-libs":
    flags = tf.sysconfig.get_link_flags()
    link_libs = []
    for flag in flags:
        if flag.startswith("-l"):
            link_libs.append(flag[2:])
    print(";".join(link_libs))
