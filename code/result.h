#ifndef RESULT_H_INCLUDED
#define RESULT_H_INCLUDED


struct Result//Result no more than a 2-value struct
{
    float score;
    int im_id;

    Result(){}

    Result(int id, float s)
    {
        this->score = s;
        this->im_id = id;
    }
    ~Result()
    {
    }

    static bool compare(Result* r1, Result* r2)
    {
        return r1->score > r2->score;
    }
};

#endif // RESULT_H_INCLUDED
