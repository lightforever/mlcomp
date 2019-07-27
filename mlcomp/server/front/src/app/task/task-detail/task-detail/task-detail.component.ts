import {Component} from '@angular/core';
import {TasksComponent} from "../../tasks/tasks.component";
import {Paginator} from "../../../paginator";
import {Task} from "../../../models";

@Component({
    selector: 'app-task-detail',
    templateUrl: './task-detail.component.html',
    styleUrls: ['../../tasks/tasks.component.css']
})
export class TaskDetailComponent extends TasksComponent{
    child_paginator: Paginator<Task>;
    id: number;

    get_filter(): any {
        let res = super.get_filter();
        res.id = this.id;
        return res;
    }

    ngOnInit() {
        super.ngOnInit();

        this.id = parseInt(this.route.snapshot.paramMap.get('id'));

         this.child_paginator = new Paginator<Task>(this.service,
                this.location,
                {
                    parent: this.id,
                    type: ['Service']
                },
                'paginator'
             );

          this.child_paginator.ngOnInit();
    }

    ngOnDestroy() {
        super.ngOnDestroy();
        this.child_paginator.ngOnDestroy();
    }

    onActivate(component) {
        component.task = this.id;
    }

}
