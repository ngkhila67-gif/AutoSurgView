// Export spelling only. The included algorithm/binding files remain byte-identical.
#if !defined(_WIN32) && !defined(__declspec)
#define __declspec(attribute) __attribute__((visibility("default")))
#endif
