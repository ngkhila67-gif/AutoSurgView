// The archived utility.cxx defines KDL::PI, but its declaration was commented
// out in utility.h. Restore only the declaration at compilation; no value or
// solver implementation is changed, and all archived files remain byte-identical.
namespace KDL { extern const double PI; }
