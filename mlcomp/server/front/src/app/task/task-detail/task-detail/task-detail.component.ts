import {Component} from '@angular/core';
import {TasksComponent} from "../../tasks/tasks.component";

@Component({
    selector: 'app-task-detail',
    templateUrl: './task-detail.component.html',
    styleUrls: ['../../tasks/tasks.component.css']
})
export class TaskDetailComponent extends TasksComponent{
    get_filter(): any {
        let res = super.get_filter();
        res.id = parseInt(this.route.snapshot.paramMap.get('id'));
        return res;
    }

    onActivate(component) {
        component.task = parseInt(this.route.snapshot.paramMap.get('id'));
    }

}
