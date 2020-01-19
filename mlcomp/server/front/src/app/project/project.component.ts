import {Component} from '@angular/core';
import {Paginator} from "../paginator";
import {Project, ProjectFilter} from "../models";
import {Location} from '@angular/common';
import {
    MatDialog,
    MatIconRegistry
} from "@angular/material";
import {DomSanitizer} from "@angular/platform-browser";
import {DialogComponent} from "../dialog/dialog.component";
import {Helpers} from "../helpers";
import {ProjectService} from "./project.service";
import {ProjectAddDialogComponent} from "./project-add-dialog";
import {AppSettings} from "../app-settings";

@Component({
    selector: 'app-project',
    templateUrl: './project.component.html',
    styleUrls: ['./project.component.css']
})
export class ProjectComponent extends Paginator<Project> {

    displayed_columns: string[] = [
        'name',
        'dag_count',
        'last_activity',
        'img_size',
        'file_size',
        'stop_all_dags',
        'remove_all_dags',
        'remove'
    ];
    name: string;
    selected: Project;

    constructor(protected service: ProjectService,
                protected location: Location,
                iconRegistry: MatIconRegistry,
                sanitizer: DomSanitizer,
                public dialog: MatDialog
    ) {
        super(service, location);
        iconRegistry.addSvgIcon('remove',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/trash.svg'));
        iconRegistry.addSvgIcon('stop',
            sanitizer.bypassSecurityTrustResourceUrl(
                'assets/img/stop.svg'));
    }

    get_filter() {
        let res = new ProjectFilter();
        res.paginator = super.get_filter();
        res.name = this.name;
        return res;
    }

    filter_name(name: string) {
        this.name = name;
        this.change.emit();
    }


    remove(element: Project) {
        const dialogRef = this.dialog.open(DialogComponent, {
            width: '550px', height: '200px',
            data: {
                'message': 'The all content will be deleted. ' +
                    'Do you want to continue?'
            }
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.remove(element.id).subscribe(data => {
                    this.change.emit();
                });
            }
        });

    }

    size(s: number) {
        return Helpers.size(s);
    }

    remove_imgs(element: Project) {
        this.service.remove_imgs(element.id).subscribe(data => {
            element.img_size = 0
        });
    }

    remove_files(element) {
        this.service.remove_files(element.id).subscribe(data => {
            element.file_size = 0
        });
    }

    stop_all_dags(element) {
        const dialogRef = this.dialog.open(DialogComponent, {
            width: '550px', height: '200px',
            data: {
                'message': 'The all content will be stopped. ' +
                    'Do you want to continue?'
            }
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.stop_all_dags(element.id).subscribe(data => {

                });
            }
        });
    }

    remove_all_dags(element) {
        const dialogRef = this.dialog.open(DialogComponent, {
            width: '550px', height: '200px',
            data: {
                'message': 'The all content will be removed. ' +
                    'Do you want to continue?'
            }
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.remove_all_dags(element.id).subscribe(data => {

                });
            }
        });
    }

    add() {
        const dialogRef = this.dialog.open(ProjectAddDialogComponent, {
            width: '600px', height: '500px',
            data: {'name': ''}
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.add(result).subscribe(_ => {
                    this.change.emit();
                });

            }
        });
    }


    edit() {
        const dialogRef = this.dialog.open(ProjectAddDialogComponent, {
            width: '600px', height: '500px',
            data: Helpers.clone(this.selected)
        });

        dialogRef.afterClosed().subscribe(result => {
            if (result) {
                this.service.edit(result).subscribe(_ => {
                    this.change.emit();
                });

            }
        });
    }
}

