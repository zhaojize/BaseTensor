#ifndef STRINGNAME_H
#define STRINGNAME_H

#include <sstream>

using namespace std;

template <class typea> bool stringname(const unsigned int &length, char* compositename, const typea & parta)
{
    stringstream ssm;
    ssm << parta <<'\0';

    string sg;

    parta >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;     
}

template <class typea, class typeb> bool stringname(const unsigned int &length, char* compositename, const typea & parta, const typeb &partb)
{
    stringstream ssm;
    ssm << parta << '_' << partb << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}

template <class typea, class typeb, class typec> bool stringname(const unsigned int &length, char* compositename, const typea & parta, 
const typeb &partb, const typec &partc)
{
    stringstream ssm;
    ssm << parta << '_' << partb << '_' << partc << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}

template <class typea, class typeb, class typec, class typed> bool stringname(const unsigned int &length, char* compositename, 
const typea & parta, const typeb &partb, const typec &partc, const typed &partd)
{
    stringstream ssm;
    ssm << parta << '_' << partb << '_' << partc << '_' << partd << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}

template <class typea, class typeb, class typec, class typed, class typee> bool stringname(const unsigned int &length, char* compositename, const typea &parta, const typeb &partb, const typec &partc, const typed &partd, const typee &parte)
{
    stringstream ssm;
    ssm << parta << '_' << partb << '_' << partc << '_' << partd << '_' << parte << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}

template <class typea, class typeb, class typec, class typed, class typee, class typef> bool stringname(const unsigned int &length, char* compositename, const typea & parta, const typeb &partb, const typec &partc, const typed &partd, const typee &parte, const typef &partf)
{
    stringstream ssm;
    ssm << parta << '_' << partb << '_' << partc << '_' << partd << '_' << parte << '_' << partf << '\0';
    
    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}



template <class typea, class typeb> bool directstringname(const unsigned int &length, char* compositename, const typea & parta, const typeb &partb)
{
    stringstream ssm;
    ssm << parta << partb << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}

template <class typea, class typeb, class typec> bool directstringname(const unsigned int &length, char* compositename, const typea & parta, 
const typeb &partb, const typec &partc)
{
    stringstream ssm;
    ssm << parta << partb << partc << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}

template <class typea, class typeb, class typec, class typed> bool directstringname(const unsigned int &length, char* compositename,
const typea & parta, const typeb &partb, const typec &partc, const typed &partd)
{
    stringstream ssm;
    ssm << parta << partb << partc << partd << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}

template <class typea, class typeb, class typec, class typed, class typee> bool directstringname(const unsigned int &length, char* compositename,
const typea & parta, const typeb &partb, const typec &partc, const typed &partd, const typee &parte)
{
    stringstream ssm;
    ssm << parta << partb << partc << partd << parte << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}

template <class typea, class typeb, class typec, class typed, class typee, class typef> bool directstringname(const unsigned int &length, char* compositename,
const typea & parta, const typeb &partb, const typec &partc, const typed &partd, const typee &parte, const typef &partf)
{
    stringstream ssm;
    ssm << parta << partb << partc << partd << parte << partf << '\0';

    string sg;

    ssm >> sg;

    if (sg.size() > length) return false;

    sg.copy(compositename, sg.size());

    return true;
}
#endif
