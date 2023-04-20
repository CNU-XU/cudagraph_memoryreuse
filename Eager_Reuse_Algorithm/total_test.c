#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

#define MAX 2147483647
#define MIN 0
//*邻接表用两部分表示  1顶点节点包含（data FirstEdge） 2边节点包含(AdjNode, *next)

typedef struct Edge{
	int AdjNode;//该弧指向顶点的位置
	struct Edge *next;//指向下一条弧的指针
} Edge;

typedef struct Node{
	int data;//顶点数据,同时也是生命周期起始点
	Edge *FirstEdge;//指向该顶点第一条边的指针
	int *adj_idx;//存储邻接点索引
	int adj_num;//邻接点数量
	int inner;//节点入度，后续用作寻找三层结构执行顺序
	int end;//生命周期结束点
	int done;//张量是否执行过算法的标志位
	int size;//张量大小
	int addr;//张量基址
	int tag;//三层结构张量基于当前张量应往上放还是往下放
} Node;

typedef struct{
	Node *node;
	int nodenum;
	int edgenum;//图的顶点数量和弧的数量
        int *idx;
} Graph;

typedef struct three_layers{
	int first;
	int intermediate;
	int last;
	struct three_layers *next;
} Three_layers;

int createGraphByFile(Graph *a,char *filename) {
    int i, j, n;
    int sum;//sum和edgenum一样代表图中总的边数
    Edge *p1, *p2;
    FILE *fp;
    int num;

    fp = fopen(filename, "r");

    printf("请输入图总顶点数和总边数\n");
    fscanf(fp,"%d %d", &a->nodenum, &a->edgenum);
    a->node = (Node *)malloc(a->nodenum * sizeof(Node));
    sum = a->edgenum;
    for(i = 0; i < a->nodenum; i++) {
        (*(a->node + i)).FirstEdge = NULL;//初始化和顶点第一个相邻的边为空;
        (*(a->node + i)).done = 0;
        (*(a->node + i)).inner = 0;
        (*(a->node + i)).tag = 0;
        (*(a->node + i)).addr = 0;
        if(i == a->nodenum - 1) {
            (*(a->node + i)).adj_idx = NULL;//为最后一个节点做初始化
            (*(a->node + i)).adj_num = 0;
        }
    }
    for(i = 0; i < a->nodenum; i++) {
        printf("***请输入第%d个顶点的值***\n", i + 1);
        fscanf(fp,"%d", &((*(a->node + i)).data));
        (*(a->node + i)).end = (*(a->node + i)).data;
        printf("请输入因该顶点产生的中间张量大小\n");
        fscanf(fp,"%d", &((*(a->node + i)).size));
        if(!sum)
            continue;
        printf("请输入以该点为起始的边数\n");
        fscanf(fp,"%d", &n);
        (*(a->node + i)).adj_idx = (int *)malloc(n * sizeof(int));
        (*(a->node + i)).adj_num = n;

        for(j = 0; j < n; j++) {
            if(!j) {
                int q;
                p1 = (Edge*)malloc(sizeof(Edge));
                p1->next = NULL;
                (*(a->node + i)).FirstEdge = p1;
                printf("请输入以该点为起始第%d条边的结束点\n", j + 1);
                fscanf(fp,"%d", &p1->AdjNode);
                *((*(a->node + i)).adj_idx + j) = p1->AdjNode;
                ((*(a->node + p1->AdjNode - 1)).inner)++;
                sum--;
            } else {
                p2 = (Edge*)malloc(sizeof(Edge));
                p2->next = NULL;
                p1->next = p2;
                printf("请输入以该点为起始第%d条边的结束点\n", j + 1);
                fscanf(fp,"%d", &p2->AdjNode);
                *((*(a->node + i)).adj_idx + j) = p2->AdjNode;
                ((*(a->node + p2->AdjNode - 1)).inner)++;
                sum--;
            }
        }
        (*(a->node + i)).end = *((*(a->node + i)).adj_idx + (*(a->node + i)).adj_num - 1);
        printf("**************************\n");
    }

    fclose(fp);
}
int createGraph(Graph *a) {
	int i, j, n;
	int sum;//sum和edgenum一样代表图中总的边数
	Edge *p1, *p2;
	printf("请输入图总顶点数和总边数\n");
	scanf("%d %d", &a->nodenum, &a->edgenum);
	a->node = (Node *)malloc(a->nodenum * sizeof(Node));
	sum = a->edgenum;
	for(i = 0; i < a->nodenum; i++) {
		(*(a->node + i)).FirstEdge = NULL;//初始化和顶点第一个相邻的边为空;
		(*(a->node + i)).done = 0;
		(*(a->node + i)).inner = 0;
		(*(a->node + i)).tag = 0;
		(*(a->node + i)).addr = 0;
		if(i == a->nodenum - 1) {
			(*(a->node + i)).adj_idx = NULL;//为最后一个节点做初始化
			(*(a->node + i)).adj_num = 0;
		}
	}
	for(i = 0; i < a->nodenum; i++) {
		printf("***请输入第%d个顶点的值***\n", i + 1);
		scanf("%d", &((*(a->node + i)).data));
		(*(a->node + i)).end = (*(a->node + i)).data;
		printf("请输入因该顶点产生的中间张量大小\n");
		scanf("%d", &((*(a->node + i)).size));
		if(!sum) 
			continue;
		printf("请输入以该点为起始的边数\n");
		scanf("%d", &n);
		(*(a->node + i)).adj_idx = (int *)malloc(n * sizeof(int));
		(*(a->node + i)).adj_num = n;

		for(j = 0; j < n; j++) {
			if(!j) {
				int q;
				p1 = (Edge*)malloc(sizeof(Edge));
				p1->next = NULL;
				(*(a->node + i)).FirstEdge = p1;
				printf("请输入以该点为起始第%d条边的结束点\n", j + 1);
				scanf("%d", &p1->AdjNode);
				*((*(a->node + i)).adj_idx + j) = p1->AdjNode;
				((*(a->node + p1->AdjNode - 1)).inner)++;
				sum--;
			} else {
				p2 = (Edge*)malloc(sizeof(Edge));
				p2->next = NULL;
				p1->next = p2;
				printf("请输入以该点为起始第%d条边的结束点\n", j + 1);
				scanf("%d", &p2->AdjNode);
				*((*(a->node + i)).adj_idx + j) = p2->AdjNode;
				((*(a->node + p2->AdjNode - 1)).inner)++;
				sum--;
			}
		}
		(*(a->node + i)).end = *((*(a->node + i)).adj_idx + (*(a->node + i)).adj_num - 1);
		printf("**************************\n");
	}
}

void print(Graph a, FILE **fp) {
	int i,j;
	for(i = 0;i < a.nodenum; i++) {
		Edge *p;
		p = (*(a.node + i)).FirstEdge;
		fprintf(*fp, " [%d]", (*(a.node + i)).data);

		while(p) {
			fprintf(*fp, "----->");
			fprintf(*fp, "[%d]", p->AdjNode);
			p = p->next;
		}
		fprintf(*fp, "\n");
	}
}

void find_structure(Graph a, Three_layers *s) {
	int i, j, k, v0, v1, v2;
	Three_layers *end = s;
	Three_layers *data;
	for(i = 0; i < a.nodenum; i++) {
		v0 = (*(a.node + i)).data;//获得first--v0
		for(j = 0; j < (*(a.node + v0 - 1)).adj_num; j++) {
			v1 = *((*(a.node + v0 - 1)).adj_idx + j);//获得Intermediate--v1
			for(k = 0; k < (*(a.node + v1 - 1)).adj_num; k++) {
				v2 = *((*(a.node + v1 - 1)).adj_idx + k);//获得last--v2
				data = (Three_layers *)malloc(sizeof(Three_layers));
				data->first = v0;
				data->intermediate = v1;
				data->last = v2;
				end->next = data;
				end = data;
			}
		}
	}
	end->next = NULL;
}

int decide_sequence(Graph *a, Three_layers **s) {
	int i = 1;
	int j;
	int node_idx;
	Three_layers *t = *s;
	Three_layers *front;
	while(1) {
		for (j = 0; j < (*(a->node + t->first - 1)).adj_num; j++){
			node_idx = *((*(a->node + t->first - 1)).adj_idx + j) - 1;
			((*(a->node + node_idx)).inner)--;
		}
		for (j = 0; j < (*(a->node + t->intermediate - 1)).adj_num; j++){
			node_idx = *((*(a->node + t->intermediate - 1)).adj_idx + j) - 1;
			((*(a->node + node_idx)).inner)--;
		}
		for (j = 0; j < (*(a->node + t->last - 1)).adj_num; j++){
			node_idx = *((*(a->node + t->last - 1)).adj_idx + j) - 1;
			((*(a->node + node_idx)).inner)--;
		}
		if ((*(a->node + t->first - 1)).inner <= 0 && (*(a->node + t->intermediate - 1)).inner <= 0
				&& (*(a->node + t->last - 1)).inner <= 0){
			return 1;
		} else {
			if (i == 1){
				front = t->next;//此处front仅为了节省空间变量重用，和front含义本身无关
				(*s)->next = front->next;
				front->next = *s;
				*s = front;
				t = *s;
				i++;
			} else {
				for(j = 0; j < i; j++){
					t = t->next;
					if (j == i - 2) {
						front = t;
					}
				}
				if (t == NULL) {
					break;
				} else {
					front->next = t->next;
					t->next = *s;
					*s = t;
				}
				i++;
			}
		}
		if (i > 20){
			printf("something wrong happen\n");
			break;
		}
	}
	return 0;
}

int find_bottom(Graph *a, int address, int idx){
	int i;
	int temp = address;
	int diff = MAX;
	address = address + (*(a->node + idx)).size;
	for (i = 0; i < a->nodenum; i++){
		if ((*(a->node + i)).done && (((*(a->node + idx)).data >= (*(a->node + i)).data
						&& (*(a->node + idx)).data <= (*(a->node + i)).end)
					|| ((*(a->node + idx)).end >= (*(a->node + i)).data
						&& (*(a->node + idx)).end <= (*(a->node + i)).end))){
			if (address > (*(a->node + i)).addr + (*(a->node + i)).size){
				if (diff > address - ((*(a->node + i)).addr + (*(a->node + i)).size)){
					diff = address - ((*(a->node + i)).addr + (*(a->node + i)).size);
					temp = (*(a->node + i)).addr + (*(a->node + i)).size;
				}
			}
		}
	}
	return temp;
}

void adjust_addr(Graph *a, int address, int shift, int idx){
	int i;
	for (i = 0; i < a->nodenum; i++){
		if ((*(a->node + i)).done && (((*(a->node + idx)).data >= (*(a->node + i)).data
						&& (*(a->node + idx)).data <= (*(a->node + i)).end)
					|| ((*(a->node + idx)).end >= (*(a->node + i)).data
						&& (*(a->node + idx)).end <= (*(a->node + i)).end))){
			if (address < (*(a->node + i)).addr){
				printf("----%d\n",(*(a->node + i)).addr);
				(*(a->node + i)).addr += shift;
			}
		}
	}
	return;
}

int solve_conflict(int address, Graph *a, int idx, int dir){
	int i;
	int flag = 0;
	int min = MAX;
	int bottom;
	if (dir == 1){
		for (i = 0; i < a->nodenum; i++){
			if ((*(a->node + i)).done && (((*(a->node + idx)).data >= (*(a->node + i)).data
							&& (*(a->node + idx)).data <= (*(a->node + i)).end)
						|| ((*(a->node + idx)).end >= (*(a->node + i)).data
							&& (*(a->node + idx)).end <= (*(a->node + i)).end))){
				if (address < (*(a->node + i)).addr + (*(a->node + i)).size){
					address = (*(a->node + i)).addr + (*(a->node + i)).size;
				}
			}
		}
		return address;
	} else if (dir == -1){
		for (i = 0; i < a->nodenum; i++){
			if ((*(a->node + i)).done && (((*(a->node + idx)).data >= (*(a->node + i)).data
							&& (*(a->node + idx)).data <= (*(a->node + i)).end)
						|| ((*(a->node + idx)).end >= (*(a->node + i)).data
							&& (*(a->node + idx)).end <= (*(a->node + i)).end))){
				if (!((address + (*(a->node + idx)).size - 1 >= (*(a->node + i)).addr + (*(a->node + i)).size)
						|| (address + (*(a->node + idx)).size - 1 < (*(a->node + i)).addr))){
					flag = 1;
				}
				if (min > (*(a->node + i)).addr){
					min = (*(a->node + i)).addr;	
				}
			}
		}
		if (flag) {
			return min - (*(a->node + idx)).size;
		} else {
			bottom = find_bottom(a, address, idx);
			if (address >= bottom){
				return address;
			} else {
				adjust_addr(a, bottom, bottom - address, idx);
				return bottom;
			}
		}
	}
}

void sort_large_first(Graph *a)
{
        int i, j;
        int *num = (int *)malloc(sizeof(int) * a->nodenum);
        a->idx = (int *)malloc(sizeof(int) * a->nodenum);
        for (i = 0; i < a->nodenum; i++){
                *(num + i) = (*(a->node + i)).size;
                *(a->idx + i) = i;
        }
        for (j = 0; j < a->nodenum; j++){
                for(i = 0; i < a->nodenum - 1 - j; i++){
                        if(*(num + i) < *(num + i + 1)){ 
                                *(num + i) = *(num + i) + *(num + i + 1);
                                *(num + i + 1) = *(num + i) - *(num + i + 1);
                                *(num + i) = *(num + i) - *(num + i + 1);
                                *(a->idx + i) = *(a->idx + i) + *(a->idx + i + 1);
                                *(a->idx + i + 1) = *(a->idx + i) - *(a->idx + i + 1);
                                *(a->idx + i) = *(a->idx + i) - *(a->idx + i + 1);
                        }
                }
        } 
        free(num);
	return;
}

void sort_short_lifetime(Graph *a)
{
        int i, j;
        int *num = (int *)malloc(sizeof(int) * a->nodenum);
        a->idx = (int *)malloc(sizeof(int) * a->nodenum);
        for (i = 0; i < a->nodenum; i++){
                *(num + i) = (*(a->node + i)).end - (*(a->node + i)).data;
                *(a->idx + i) = i;
        }
        for (j = 0; j < a->nodenum; j++){
                for(i = 0; i < a->nodenum - 1 - j; i++){
                        if(*(num + i) > *(num + i + 1)){ 
                                *(num + i) = *(num + i) + *(num + i + 1);
                                *(num + i + 1) = *(num + i) - *(num + i + 1);
                                *(num + i) = *(num + i) - *(num + i + 1);
                                *(a->idx + i) = *(a->idx + i) + *(a->idx + i + 1);
                                *(a->idx + i + 1) = *(a->idx + i) - *(a->idx + i + 1);
                                *(a->idx + i) = *(a->idx + i) - *(a->idx + i + 1);
                        }
                }
        } 
        free(num);
	return;
}

void eager_reuse(Graph *a, Three_layers *s){
	int i;
	int *t = (int *)malloc(4 * sizeof(int));
	int temp_addr;
	if ((*(a->node + s->first - 1)).tag != 0){
		(*(a->node + s->intermediate - 1)).tag = -1 * (*(a->node + s->first - 1)).tag;
		(*(a->node + s->last - 1)).tag = (*(a->node + s->first - 1)).tag;
	} else if ((*(a->node + s->intermediate - 1)).tag != 0){
		(*(a->node + s->first - 1)).tag = -1 * (*(a->node + s->intermediate - 1)).tag;
		(*(a->node + s->last - 1)).tag = -1 * (*(a->node + s->intermediate - 1)).tag;
	} else if ((*(a->node + s->last - 1)).tag != 0){
		(*(a->node + s->first - 1)).tag = (*(a->node + s->last - 1)).tag;
		(*(a->node + s->intermediate - 1)).tag = -1 * (*(a->node + s->last - 1)).tag;
	} else {
		(*(a->node + s->first - 1)).tag = 1;
		(*(a->node + s->intermediate - 1)).tag = -1;
		(*(a->node + s->last - 1)).tag = 1;
		(*(a->node + s->first - 1)).addr = 0;
		(*(a->node + s->first - 1)).done = 1;
	}
	if ((*(a->node + s->first - 1)).done){
		*(t + 0) = s->intermediate - 1;
		*(t + 1) = s->first - 1;
		*(t + 2) = s->last - 1;
		*(t + 3) = s->intermediate - 1;
	} else if ((*(a->node + s->intermediate - 1)).done){
		*(t + 0) = s->first - 1;
		*(t + 1) = s->intermediate - 1;
		*(t + 2) = s->last - 1;	
		*(t + 3) = s->intermediate - 1;
	} else if ((*(a->node + s->last - 1)).done){
		*(t + 0) = s->intermediate - 1;
		*(t + 1) = s->last - 1;
		*(t + 2) = s->first - 1;	
		*(t + 3) = s->intermediate - 1;
	} else {
		printf("something wrong happen\n");
		return;
	}
	for (i = 0; i < 4; i += 2){
		if ((*(a->node + *(t + i))).done == 0){
			if ((*(a->node + *(t + i + 1))).tag == 1){
				temp_addr = (*(a->node + *(t + i + 1))).addr + (*(a->node + *(t + i + 1))).size;
				(*(a->node + *(t + i))).addr = solve_conflict(temp_addr, a, *(t + i), 1);
			} else {
				temp_addr = (*(a->node + *(t + i + 1))).addr - (*(a->node + *(t + i))).size;
				(*(a->node + *(t + i))).addr = solve_conflict(temp_addr, a, *(t + i), -1);
			}
			(*(a->node + *(t + i))).done = 1;
		}
	}
	free(t);
	return;
}

int TfLite_memory_reuse(Graph *a, int step){
	int i, j, flag;
	int max = MIN;
	int min = MAX;
	int *p;
	int count = 0;
	if (step != 0) {
		for (i = 0; i < step; i++){
			if (((*(a->node + *(a->idx + i))).data <= (*(a->node + *(a->idx + step))).data
						&& (*(a->node + *(a->idx + i))).end >= (*(a->node + *(a->idx + step))).data)
					|| ((*(a->node + *(a->idx + i))).data <= (*(a->node + *(a->idx + step))).end
						&& (*(a->node + *(a->idx + i))).end >= (*(a->node + *(a->idx + step))).end)){
				if (max < (*(a->node + *(a->idx + i))).addr + (*(a->node + *(a->idx + i))).size){
					max = (*(a->node + *(a->idx + i))).addr + (*(a->node + *(a->idx + i))).size;
				}
				if (min > (*(a->node + *(a->idx + i))).addr){
					min = (*(a->node + *(a->idx + i))).addr;
				}
				count++;
			}
		}
		if (count > 1 && (max - min) >= (*(a->node + *(a->idx + step))).size){
			p = (int *)calloc(count, sizeof(int));
			j = 0;
			for (i = 0; i < step; i++){
				if (((*(a->node + *(a->idx + i))).data <= (*(a->node + *(a->idx + step))).data
							&& (*(a->node + *(a->idx + i))).end >= (*(a->node + *(a->idx + step))).data)
						|| ((*(a->node + *(a->idx + i))).data <= (*(a->node + *(a->idx + step))).end
							&& (*(a->node + *(a->idx + i))).end >= (*(a->node + *(a->idx + step))).end)){
					*(p + j++) = i;
				}
			}
		} else {
			goto final;
		}
		int temp0, temp1;
		for (i = count - 1; i >= 0; i--){
			flag = 1;
			temp0 = (*(a->node + *(a->idx + *(p + i)))).addr + (*(a->node + *(a->idx + *(p + i)))).size;
			temp1 = temp0 + (*(a->node + *(a->idx + step))).size;
			for (j = count - 1; j >= 0; j--){
				if (temp0 > (*(a->node + *(a->idx + *(p + j)))).addr
						&& temp0 < (*(a->node + *(a->idx + *(p + j)))).addr + (*(a->node + *(a->idx + *(p + j)))).size){
					flag = 0;
				}
				if (temp1 > (*(a->node + *(a->idx + *(p + j)))).addr
						&& temp1 < (*(a->node + *(a->idx + *(p + j)))).addr + (*(a->node + *(a->idx + *(p + j)))).size){
					flag = 0;
				}
				if (temp0 == max){
					flag = 0;
				}
			}
			if (flag){
				max = temp0;
				break;
			}
		}
	}
final:
	(*(a->node + *(a->idx + step))).addr = max;
	return max + (*(a->node + *(a->idx + step))).size;
}

int large_tensor_first(Graph *a, int step){
        int i;
        int max = 0;
        for (i = 0; i < step; i++){
                if (((*(a->node + *(a->idx + i))).data <= (*(a->node + *(a->idx + step))).data
                                && (*(a->node + *(a->idx + i))).end >= (*(a->node + *(a->idx + step))).data)
                                || ((*(a->node + *(a->idx + i))).data <= (*(a->node + *(a->idx + step))).end
                                        && (*(a->node + *(a->idx + i))).end >= (*(a->node + *(a->idx + step))).end)){
                        if (max < (*(a->node + *(a->idx + i))).addr + (*(a->node + *(a->idx + i))).size){
                                max = (*(a->node + *(a->idx + i))).addr + (*(a->node + *(a->idx + i))).size;
                        }
                }
        }
        (*(a->node + *(a->idx + step))).addr = max;
        return max + (*(a->node + *(a->idx + step))).size;
}

int short_time_first(Graph *a, int step){
        int i;
        int max = 0;
        for (i = 0; i < step; i++){
                if (((*(a->node + *(a->idx + i))).data <= (*(a->node + *(a->idx + step))).data
                                && (*(a->node + *(a->idx + i))).end >= (*(a->node + *(a->idx + step))).data)
                                || ((*(a->node + *(a->idx + i))).data <= (*(a->node + *(a->idx + step))).end
                                        && (*(a->node + *(a->idx + i))).end >= (*(a->node + *(a->idx + step))).end)){
                        if (max < (*(a->node + *(a->idx + i))).addr + (*(a->node + *(a->idx + i))).size){
                                max = (*(a->node + *(a->idx + i))).addr + (*(a->node + *(a->idx + i))).size;
                        }
                }
        }
        (*(a->node + *(a->idx + step))).addr = max;
        return max + (*(a->node + *(a->idx + step))).size;
}

void test1(Graph g, FILE **fp){
	int i;
	int min = 0;
	int max = 0;
	struct timeval begin, end;
	Three_layers *structure, *temp, *front;
	//printf("***三层结构结果输出***\n");
	structure = (Three_layers *)malloc(sizeof(Three_layers));
	structure->next = NULL;//初始化三层结构存储链表
	find_structure(g, structure);
	front = structure;
	temp = structure->next;
	fprintf(*fp, "***及时重用算法执行***\n");
	gettimeofday(&begin, NULL);
	while(temp) {
		if ((!((*(g.node + temp->first - 1)).done * (*(g.node + temp->intermediate - 1)).done
						* (*(g.node + temp->last - 1)).done)) && decide_sequence(&g, &temp)){
			//printf("[%d-%d-%d]\n", temp->first, temp->intermediate, temp->last);
			eager_reuse(&g, temp);//此处进行三层结构执行
			if (min > (*(g.node + temp->first - 1)).addr){
				min = (*(g.node + temp->first - 1)).addr;
			}
			if (min > (*(g.node + temp->intermediate - 1)).addr){
				min = (*(g.node + temp->intermediate - 1)).addr;
			}
			if (min > (*(g.node + temp->last - 1)).addr){
				min = (*(g.node + temp->last - 1)).addr;
			}
			if (max < (*(g.node + temp->first - 1)).addr + (*(g.node + temp->first - 1)).size){
				max = (*(g.node + temp->first - 1)).addr + (*(g.node + temp->first - 1)).size;
			}
			if (max < (*(g.node + temp->intermediate - 1)).addr + (*(g.node + temp->intermediate - 1)).size){
				max = (*(g.node + temp->intermediate - 1)).addr + (*(g.node + temp->intermediate - 1)).size;
			}
			if (max < (*(g.node + temp->last - 1)).addr + (*(g.node + temp->last - 1)).size){
				max = (*(g.node + temp->last - 1)).addr + (*(g.node + temp->last - 1)).size;
			}
		}
		front->next = temp;
		front = temp;
		temp = temp->next;
	}
	gettimeofday(&end, NULL);
	for (i = 0; i < g.nodenum; i++){
		(*(g.node + i)).addr -= min;
		fprintf(*fp, "tensor[%d]--------base[%d]--------top[%d]--------begin[%d]--------end[%d]\n",
				i + 1, (*(g.node + i)).addr, (*(g.node + i)).addr + (*(g.node + i)).size,
				(*(g.node + i)).data, (*(g.node + i)).end);	
	}
	fprintf(*fp, "eager_reuse:\t\tmemory_size=%d\toverhead=%.3fms\n", max - min,
			(end.tv_sec - begin.tv_sec) * 1000.0 + (end.tv_usec - begin.tv_usec) / 1000.0);
}

void test2(Graph g, FILE **fp){
        int memory_size = 0;
        int this_top;
        int i;
	struct timeval begin, end;
        sort_large_first(&g);
        fprintf(*fp, "***TfLite内存重用算法执行***\n");
	gettimeofday(&begin, NULL);
        for (i = 0; i < g.nodenum; i++){
                this_top = TfLite_memory_reuse(&g, i);
                if (memory_size < this_top){
                        memory_size = this_top;
                }
        }
	gettimeofday(&end, NULL);
	for (i = 0; i < g.nodenum; i++){
		fprintf(*fp, "tensor[%d]--------base[%d]--------top[%d]--------begin[%d]--------end[%d]\n",
				i + 1, (*(g.node + i)).addr, (*(g.node + i)).addr + (*(g.node + i)).size,
				(*(g.node + i)).data, (*(g.node + i)).end);	
	}
        fprintf(*fp, "TfLite_memory_reuse:\tmemory_size=%d\toverhead=%.3fms\n", memory_size,
			(end.tv_sec - begin.tv_sec) * 1000.0 + (end.tv_usec - begin.tv_usec) / 1000.0);
}

void test3(Graph g, FILE **fp){
        int memory_size = 0;
        int this_top;
        int i;
	struct timeval begin, end;
        sort_large_first(&g);
        fprintf(*fp, "***大张量优先算法执行***\n");
	gettimeofday(&begin, NULL);
        for (i = 0; i < g.nodenum; i++){
                this_top = large_tensor_first(&g, i);
                if (memory_size < this_top){
                        memory_size = this_top;
                }
        }
	gettimeofday(&end, NULL);
	for (i = 0; i < g.nodenum; i++){
		fprintf(*fp, "tensor[%d]--------base[%d]--------top[%d]--------begin[%d]--------end[%d]\n",
				i + 1, (*(g.node + i)).addr, (*(g.node + i)).addr + (*(g.node + i)).size,
				(*(g.node + i)).data, (*(g.node + i)).end);	
	}
        fprintf(*fp, "large_tensor_first:\tmemory_size=%d\toverhead=%.3fms\n", memory_size,
			(end.tv_sec - begin.tv_sec) * 1000.0 + (end.tv_usec - begin.tv_usec) / 1000.0);
}

void test4(Graph g, FILE **fp){
        int memory_size = 0;
        int this_top;
        int i;
	struct timeval begin, end;
        sort_short_lifetime(&g);
        fprintf(*fp, "***短生命周期优先算法执行***\n");
	gettimeofday(&begin, NULL);
        for (i = 0; i < g.nodenum; i++){
                this_top = short_time_first(&g, i);
                if (memory_size < this_top){
                        memory_size = this_top;
                }
        }
	gettimeofday(&end, NULL);
	for (i = 0; i < g.nodenum; i++){
		fprintf(*fp, "tensor[%d]--------base[%d]--------top[%d]--------begin[%d]--------end[%d]\n",
				i + 1, (*(g.node + i)).addr, (*(g.node + i)).addr + (*(g.node + i)).size,
				(*(g.node + i)).data, (*(g.node + i)).end);	
	}
        fprintf(*fp, "short_time_first:\tmemory_size=%d\toverhead=%.3fms\n", memory_size,
			(end.tv_sec - begin.tv_sec) * 1000.0 + (end.tv_usec - begin.tv_usec) / 1000.0);
}

int main(int argc,char** argv) 
{   
	Graph g;
	char *routine = (char *)calloc(1, 256);
	FILE *fp;
	if (argc < 2){
		printf("no input parameter!\n");
		exit(0);
	} else{
		sprintf(routine, "result/%s", argv[1]);
		if ((fp = fopen(routine, "w")) == NULL){
			printf("open file error!\n");
			exit(0);
		}
	}
    int choice;
    printf("文件输入图请输入1，手动输入图请输入2： ");
    scanf("%d",&choice);
    char filename[100];
    switch(choice){
    case 1: printf("请输入输入文件名：") ;
        scanf("%s",filename);
        createGraphByFile(&g,filename);break;
    case 2:
        createGraph(&g);}
	fprintf(fp, "***邻接表形式输出图***\n");
	print(g, &fp);//邻接表形式输出图
	test1(g, &fp);//eager_reuse
	test2(g, &fp);//TfLite_memory_reuse
	test3(g, &fp);//large_tensor_first
	test4(g, &fp);//short_time_first
	fclose(fp);
	return 0;
}
