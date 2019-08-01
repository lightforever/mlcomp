import {Component, Inject} from "@angular/core";
import {MAT_DIALOG_DATA} from "@angular/material";
import {TaskService} from "../task.service";
import {TaskInfo} from "../../models";

@Component({
    selector: 'task-info-dialog.component.css',
    templateUrl: 'task-info-dialog.component.html',
})
export class TaskInfoDialogComponent {
    error: string;

    constructor(
        @Inject(MAT_DIALOG_DATA) public data: TaskInfo,
        public service: TaskService) {

        this.service.info(data.id).subscribe(res => {
            this.data = res;
        });
    }

}